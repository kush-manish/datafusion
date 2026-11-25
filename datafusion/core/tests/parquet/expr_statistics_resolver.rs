// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! A [`ParquetStatisticsResolver`] registered from outside DataFusion prunes
//! row groups for its own function.
//!
//! `fake_variant_get(v, 'a')` stands in for a shredded variant access. The
//! file stores `a` in `v.typed_value.a.typed_value` when it is shredded and
//! in `v.typed_value.a.value` (binary) when it is not:
//!
//! | Row group | `typed_value` | `value`      | Bounds usable? |
//! |-----------|---------------|--------------|----------------|
//! | 0         | 1, 2          | null, null   | yes: [1, 2]    |
//! | 1         | 10, 11        | null, null   | yes: [10, 11]  |
//! | 2         | 3, null       | null, <100>  | no: one row is not shredded |
//!
//! For `fake_variant_get(v, 'a') > 5`, only row group 0 can be pruned.
//!
//! The resolver does not implement `bloom_filter_leaf` or `page_index_leaf`.
//! A bloom filter on `typed_value` cannot see values stored in `value`, and
//! its validity check needs a second leaf, whose pages do not line up.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Int64Array, RecordBatch, StructArray,
    UInt64Array,
};
use arrow::compute::nullif;
use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::ParquetSource;
use datafusion::prelude::SessionContext;
use datafusion_common::{DFSchema, Result, ScalarValue, internal_err};
use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource_parquet::statistics_resolver::{
    ExprStatisticsKind, ParquetLeafStatistics, ParquetStatisticsResolver,
    leaf_index_by_path,
};
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
    col, lit,
};
use datafusion_physical_expr::expressions::{Column, Literal};
use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::metrics::MetricValue;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::NamedTempFile;

use crate::parquet::utils::MetricsFinder;

/// Returns `v.typed_value.<key>`. Only its statistics are used here.
#[derive(Debug, PartialEq, Eq, Hash)]
struct FakeVariantGet(Signature);

impl ScalarUDFImpl for FakeVariantGet {
    fn name(&self) -> &str {
        "fake_variant_get"
    }

    fn signature(&self) -> &Signature {
        &self.0
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int64)
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        internal_err!("the scan only prunes with statistics, it does not filter rows")
    }
}

/// Reads `fake_variant_get(v, 'key')` bounds from the shredded column, but
/// only for containers where no row fell back to the binary `value` column.
#[derive(Debug)]
struct FakeVariantResolver;

impl FakeVariantResolver {
    /// The column name and key of `fake_variant_get(column, 'key')`.
    fn args(expr: &Arc<dyn PhysicalExpr>) -> Option<(&str, &str)> {
        let func = expr.downcast_ref::<ScalarFunctionExpr>()?;
        if func.name() != "fake_variant_get" {
            return None;
        }
        let [column, key] = func.args() else {
            return None;
        };
        let column = column.downcast_ref::<Column>()?;
        let ScalarValue::Utf8(Some(key)) = key.downcast_ref::<Literal>()?.value() else {
            return None;
        };
        Some((column.name(), key.as_str()))
    }
}

impl ParquetStatisticsResolver for FakeVariantResolver {
    fn matches(&self, expr: &Arc<dyn PhysicalExpr>) -> bool {
        Self::args(expr).is_some()
    }

    fn statistics(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        kind: ExprStatisticsKind,
        statistics: &dyn ParquetLeafStatistics,
    ) -> Option<ArrayRef> {
        let (column, key) = Self::args(expr)?;
        let parquet_schema = statistics.parquet_schema();
        let typed = leaf_index_by_path(
            parquet_schema,
            &[column, "typed_value", key, "typed_value"],
        )?;
        let fallback =
            leaf_index_by_path(parquet_schema, &[column, "typed_value", key, "value"])?;
        let typed_field = Field::new("typed_value", DataType::Int64, true);
        let fallback_field = Field::new("value", DataType::Binary, true);

        let bounds = match kind {
            ExprStatisticsKind::Min => statistics.leaf_mins(typed, &typed_field)?,
            ExprStatisticsKind::Max => statistics.leaf_maxes(typed, &typed_field)?,
            // A null `typed_value` may hold a value in `value`.
            ExprStatisticsKind::NullCount => return None,
        };
        // The bounds hold only where every row is shredded, which is where
        // the fallback column is null in every row.
        let fallback_nulls = statistics.leaf_null_counts(fallback, &fallback_field)?;
        let rows = statistics.row_counts()?;
        let fallback_nulls = fallback_nulls.as_any().downcast_ref::<UInt64Array>()?;
        let rows = rows.as_any().downcast_ref::<UInt64Array>()?;
        // Unknown counts are null. Treat them as not shredded.
        let not_shredded: BooleanArray = fallback_nulls
            .iter()
            .zip(rows.iter())
            .map(|(nulls, rows)| Some(nulls.is_none() || nulls != rows))
            .collect();
        nullif(&bounds, &not_shredded).ok()
    }
}

fn shredded_schema() -> SchemaRef {
    let a = Fields::from(vec![
        Field::new("value", DataType::Binary, true),
        Field::new("typed_value", DataType::Int64, true),
    ]);
    let typed_value = Fields::from(vec![Field::new("a", DataType::Struct(a), true)]);
    let v = Fields::from(vec![Field::new(
        "typed_value",
        DataType::Struct(typed_value),
        true,
    )]);
    Arc::new(Schema::new(vec![Field::new(
        "v",
        DataType::Struct(v),
        true,
    )]))
}

/// Writes the file described in the module docs, two rows per row group.
fn write_file(schema: &SchemaRef) -> NamedTempFile {
    let DataType::Struct(v_fields) = schema.field(0).data_type() else {
        unreachable!()
    };
    let DataType::Struct(tv_fields) = v_fields[0].data_type() else {
        unreachable!()
    };
    let DataType::Struct(a_fields) = tv_fields[0].data_type() else {
        unreachable!()
    };
    let value: ArrayRef = Arc::new(BinaryArray::from(vec![
        None,
        None,
        None,
        None,
        None,
        Some(100i64.to_le_bytes().as_slice()),
    ]));
    let typed: ArrayRef = Arc::new(Int64Array::from(vec![
        Some(1),
        Some(2),
        Some(10),
        Some(11),
        Some(3),
        None,
    ]));
    let a = StructArray::new(a_fields.clone(), vec![value, typed], None);
    let tv = StructArray::new(tv_fields.clone(), vec![Arc::new(a)], None);
    let v = StructArray::new(v_fields.clone(), vec![Arc::new(tv)], None);
    let batch = RecordBatch::try_new(Arc::clone(schema), vec![Arc::new(v)]).unwrap();

    let file = NamedTempFile::new().unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2))
        .build();
    let mut writer =
        ArrowWriter::try_new(file.reopen().unwrap(), Arc::clone(schema), Some(props))
            .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    file
}

/// Scans the file with `fake_variant_get(v, 'a') > 5` and returns the
/// (pruned, matched) row group counts from statistics.
async fn row_groups_pruned(register_resolver: bool) -> (usize, usize) {
    let schema = shredded_schema();
    let file = write_file(&schema);
    let ctx = SessionContext::new();

    let udf = ScalarUDF::from(FakeVariantGet(Signature::any(2, Volatility::Immutable)));
    let predicate = udf.call(vec![col("v"), lit("a")]).gt(lit(5i64));
    let df_schema = DFSchema::try_from(Arc::clone(&schema)).unwrap();
    let predicate = ctx.create_physical_expr(predicate, &df_schema).unwrap();

    let mut source = ParquetSource::new(Arc::clone(&schema)).with_predicate(predicate);
    if register_resolver {
        source = source.with_statistics_resolver(Arc::new(FakeVariantResolver));
    }
    let path = file.path().to_str().unwrap().to_string();
    let size = std::fs::metadata(file.path()).unwrap().len();
    let config =
        FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), Arc::new(source))
            .with_file(PartitionedFile::new(path, size))
            .build();
    let plan: Arc<dyn ExecutionPlan> = DataSourceExec::from_data_source(config);
    datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx())
        .await
        .unwrap();

    let metrics = MetricsFinder::find_metrics(plan.as_ref()).unwrap();
    let Some(MetricValue::PruningMetrics {
        pruning_metrics, ..
    }) = metrics.sum_by_name("row_groups_pruned_statistics")
    else {
        panic!("metrics: {metrics:#?}");
    };
    (pruning_metrics.pruned(), pruning_metrics.matched())
}

#[tokio::test]
async fn external_resolver_prunes_shredded_row_groups() {
    // Row group 0 is pruned. Row group 2 is kept because one of its rows
    // is not shredded, even though its shredded max (3) is below 5.
    assert_eq!(row_groups_pruned(true).await, (1, 2));
}

#[tokio::test]
async fn without_resolver_nothing_is_pruned() {
    assert_eq!(row_groups_pruned(false).await, (0, 3));
}
