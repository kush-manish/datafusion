// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Statistics for expressions, such as struct field and variant accesses,
//! read from Parquet leaf column statistics.
//!
//! ```text
//! PruningPredicate                         Parquet scan
//!   asks the matcher: is `expr` a key? --> ParquetStatisticsResolvers::matches
//!   min_values_for_expr(expr) ----------> RowGroupPruningStatistics / PagesPruningStatistics
//!                                            -> each resolver that matches `expr`
//!                                               reads leaf statistics through
//!                                               ParquetLeafStatistics
//! ```
//!
//! A resolver works on [`ParquetLeafStatistics`], so the same resolver serves
//! row group statistics and the page index.

use std::fmt::Debug;
use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_common::ScalarValue;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_physical_expr::expressions::{Column, Literal};
use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion_pruning::StatisticsExprMatcher;
use parquet::schema::types::SchemaDescriptor;

/// Which statistic a [`ParquetStatisticsResolver`] is asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExprStatisticsKind {
    /// Lower bounds, one per container.
    Min,
    /// Upper bounds, one per container.
    Max,
    /// The number of rows where the expression is null, one per container.
    NullCount,
}

/// Statistics of Parquet leaf columns for a set of containers: the row
/// groups of a file, or the pages of one row group.
pub trait ParquetLeafStatistics {
    /// The Arrow schema of the file. Expressions refer to its columns.
    fn arrow_schema(&self) -> &Schema;

    /// The Parquet schema of the file.
    fn parquet_schema(&self) -> &SchemaDescriptor;

    /// The number of containers.
    fn num_containers(&self) -> usize;

    /// Minimum values of the Parquet leaf column `leaf`, converted to the
    /// type of `field`. Returns `None` when the statistics are unknown or
    /// cannot be trusted.
    fn leaf_mins(&self, leaf: usize, field: &Field) -> Option<ArrayRef>;

    /// Maximum values of the Parquet leaf column `leaf`. See
    /// [`Self::leaf_mins`].
    fn leaf_maxes(&self, leaf: usize, field: &Field) -> Option<ArrayRef>;

    /// Null counts of the Parquet leaf column `leaf`, as a `UInt64Array`.
    /// For a leaf with no repeated ancestor, a row counts as null when the
    /// leaf or any ancestor is null.
    fn leaf_null_counts(&self, leaf: usize, field: &Field) -> Option<ArrayRef>;

    /// Row counts, as a `UInt64Array`.
    fn row_counts(&self) -> Option<ArrayRef>;
}

/// Reads statistics for an expression from Parquet leaf statistics.
///
/// DataFusion provides [`GetFieldStatisticsResolver`] for struct fields.
/// Other crates can add resolvers for their own functions, for example a
/// variant extension that reads the statistics of shredded `typed_value`
/// columns for `variant_get`. Register them with
/// [`ParquetSource::with_statistics_resolver`].
///
/// [`ParquetSource::with_statistics_resolver`]: crate::source::ParquetSource::with_statistics_resolver
pub trait ParquetStatisticsResolver: Debug + Send + Sync {
    /// Returns `true` if this resolver may have statistics for `expr`.
    ///
    /// This runs while pruning predicates are built, before any file
    /// metadata is known, so it should only look at the shape of `expr`.
    fn matches(&self, expr: &Arc<dyn PhysicalExpr>) -> bool;

    /// Returns one value per container for `kind`, or `None` if this
    /// resolver cannot provide statistics for `expr` in this file.
    ///
    /// Bounds must be conservative: each non-null min (max) must be at most
    /// (at least) every non-null value of `expr` in its container. Return
    /// null for a container whose bounds are unknown.
    fn statistics(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        kind: ExprStatisticsKind,
        statistics: &dyn ParquetLeafStatistics,
    ) -> Option<ArrayRef>;

    /// The Parquet leaf whose pages hold the values of `expr`, if page
    /// index pruning can use this resolver.
    ///
    /// The page index prunes one leaf at a time, because the pages of
    /// different leaves do not line up. Return `None` (the default) when the
    /// statistics need more than one leaf.
    fn page_index_leaf(
        &self,
        _expr: &Arc<dyn PhysicalExpr>,
        _arrow_schema: &Schema,
        _parquet_schema: &SchemaDescriptor,
    ) -> Option<usize> {
        None
    }
}

/// The [`ParquetStatisticsResolver`]s of a Parquet scan.
///
/// The default holds [`GetFieldStatisticsResolver`]. It is also the
/// [`StatisticsExprMatcher`] given to pruning predicates, so a predicate
/// only treats an expression as a statistics key if a resolver may answer it.
#[derive(Debug, Clone)]
pub struct ParquetStatisticsResolvers {
    resolvers: Vec<Arc<dyn ParquetStatisticsResolver>>,
}

/// Resolvers are equal when they hold the same resolver instances.
impl PartialEq for ParquetStatisticsResolvers {
    fn eq(&self, other: &Self) -> bool {
        self.resolvers.len() == other.resolvers.len()
            && self
                .resolvers
                .iter()
                .zip(&other.resolvers)
                .all(|(a, b)| Arc::ptr_eq(a, b))
    }
}

impl Default for ParquetStatisticsResolvers {
    fn default() -> Self {
        Self {
            resolvers: vec![Arc::new(GetFieldStatisticsResolver)],
        }
    }
}

impl ParquetStatisticsResolvers {
    /// Adds `resolver`. Resolvers are asked in the order they were added.
    pub fn with_resolver(mut self, resolver: Arc<dyn ParquetStatisticsResolver>) -> Self {
        self.resolvers.push(resolver);
        self
    }

    /// Returns `true` if resolvers beyond the built-in ones were added.
    #[cfg(feature = "proto")]
    pub(crate) fn has_custom_resolvers(&self) -> bool {
        self.resolvers.len() > Self::default().resolvers.len()
    }

    /// The first statistics that a matching resolver provides.
    pub(crate) fn statistics(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        kind: ExprStatisticsKind,
        statistics: &dyn ParquetLeafStatistics,
    ) -> Option<ArrayRef> {
        self.resolvers
            .iter()
            .filter(|r| r.matches(expr))
            .find_map(|r| r.statistics(expr, kind, statistics))
    }

    /// The page index leaf that the first matching resolver reports.
    pub(crate) fn page_index_leaf(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        arrow_schema: &Schema,
        parquet_schema: &SchemaDescriptor,
    ) -> Option<usize> {
        self.resolvers
            .iter()
            .filter(|r| r.matches(expr))
            .find_map(|r| r.page_index_leaf(expr, arrow_schema, parquet_schema))
    }
}

impl StatisticsExprMatcher for ParquetStatisticsResolvers {
    fn matches(&self, expr: &Arc<dyn PhysicalExpr>) -> bool {
        self.resolvers.iter().any(|r| r.matches(expr))
    }
}

/// Returns the index of the Parquet leaf column at `path`, such as
/// `["s", "a"]` for the field `a` of the struct column `s`.
pub fn leaf_index_by_path(
    parquet_schema: &SchemaDescriptor,
    path: &[&str],
) -> Option<usize> {
    parquet_schema.columns().iter().position(|descr| {
        let parts = descr.path().parts();
        parts.len() == path.len() && parts.iter().zip(path).all(|(a, b)| a == b)
    })
}

/// Reads the statistics of `get_field(column, 'a', 'b', ...)` from the
/// Parquet leaf of that struct field.
///
/// Only struct paths are supported. A path through a map is a runtime key
/// lookup, and a field below a list has statistics per element, not per row.
#[derive(Debug, Default)]
pub struct GetFieldStatisticsResolver;

impl GetFieldStatisticsResolver {
    /// The root column and field path of `expr`, including chained calls
    /// such as `get_field(get_field(s, 'a'), 'b')`.
    fn path(expr: &Arc<dyn PhysicalExpr>) -> Option<(&Column, Vec<&str>)> {
        let func = ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(expr.as_ref())?;
        let (base, keys) = func.args().split_first()?;
        let (column, mut path) = match base.downcast_ref::<Column>() {
            Some(column) => (column, vec![]),
            None => Self::path(base)?,
        };
        for key in keys {
            match key.downcast_ref::<Literal>()?.value() {
                ScalarValue::Utf8(Some(name))
                | ScalarValue::LargeUtf8(Some(name))
                | ScalarValue::Utf8View(Some(name)) => path.push(name.as_str()),
                _ => return None,
            }
        }
        (!path.is_empty()).then_some((column, path))
    }

    /// The Parquet leaf index and Arrow field of `expr`.
    fn leaf<'a>(
        expr: &Arc<dyn PhysicalExpr>,
        arrow_schema: &'a Schema,
        parquet_schema: &SchemaDescriptor,
    ) -> Option<(usize, &'a Field)> {
        let (column, path) = Self::path(expr)?;
        let mut field = arrow_schema.field_with_name(column.name()).ok()?;
        for name in &path {
            let DataType::Struct(fields) = field.data_type() else {
                return None;
            };
            field = fields.iter().find(|f| f.name() == name)?;
        }
        if field.data_type().is_nested() {
            return None;
        }
        let mut parts = vec![column.name()];
        parts.extend(path);
        let leaf = leaf_index_by_path(parquet_schema, &parts)?;
        (parquet_schema.column(leaf).max_rep_level() == 0).then_some((leaf, field))
    }
}

impl ParquetStatisticsResolver for GetFieldStatisticsResolver {
    fn matches(&self, expr: &Arc<dyn PhysicalExpr>) -> bool {
        Self::path(expr).is_some()
    }

    fn statistics(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        kind: ExprStatisticsKind,
        statistics: &dyn ParquetLeafStatistics,
    ) -> Option<ArrayRef> {
        let (leaf, field) =
            Self::leaf(expr, statistics.arrow_schema(), statistics.parquet_schema())?;
        match kind {
            ExprStatisticsKind::Min => statistics.leaf_mins(leaf, field),
            ExprStatisticsKind::Max => statistics.leaf_maxes(leaf, field),
            ExprStatisticsKind::NullCount => statistics.leaf_null_counts(leaf, field),
        }
    }

    fn page_index_leaf(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        arrow_schema: &Schema,
        parquet_schema: &SchemaDescriptor,
    ) -> Option<usize> {
        Self::leaf(expr, arrow_schema, parquet_schema).map(|(leaf, _)| leaf)
    }
}
