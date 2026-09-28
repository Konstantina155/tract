use crate::internal::*;
use crate::ser::*;
use tract_core::ops::matmul::lir_unary::{
    AddMatMulGeometry, LirMatMulUnary, MapOutputAxisToInput, ProtoFusedSpec,
};
use tract_linalg::mmm::BinOp;
use tract_linalg::mmm::{RoundingPolicy, InputStoreSpec, OutputStoreSpec, PrepackedSpec};
use tract_core::tract_linalg::mmm::MatMatMulKer;
use crate::ast::Literal;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_lir_matmul_unary);

    registry.register_primitive(
        "tract_core_lir_matmul_unary",
        &parameters(),
        &[("output", TypeName::Scalar.tensor())],
        de_lir_matmul_unary,
    );
}

fn parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().array().named("inputs"),
        TypeName::Integer.named("c_m_axis"),
        TypeName::Integer.named("c_n_axis"),
        TypeName::String.named("kernel"),
        TypeName::String.named("c_datum_type"),
        TypeName::Any.array().named("c_shape"),
        TypeName::Any.array().named("micro_ops"),
    ]
}

fn ser_lir_matmul_unary(ast: &mut IntoAst, node: &TypedNode, op: &LirMatMulUnary) -> TractResult<Option<Arc<RValue>>> {
    let inputs: Vec<_> = node.inputs.iter().map(|i| (*ast.mapping[i]).clone()).collect();

    let micro_ops: Vec<RValue> =
        op.micro_ops.iter().map(serialize_proto_fused_spec).collect::<TractResult<_>>()?;

    Ok(Some(invocation(
        "tract_core_lir_matmul_unary",
        &[Arc::new(RValue::Array(inputs))],
        &[
            ("c_m_axis", numeric(op.c_m_axis)),
            ("c_n_axis", numeric(op.c_n_axis)),
            ("kernel", string(op.mmm.kernel_name())),
            ("c_datum_type", string(format!("{:?}", op.c_fact.datum_type))),
            ("c_shape", array(&op.c_fact.shape.iter().map(serialize_tdim).collect::<Vec<_>>())),
            ("micro_ops", array(&micro_ops)),
        ],
    )))
}

fn serialize_proto_fused_spec(spec: &ProtoFusedSpec) -> TractResult<RValue> {
    Ok(match spec {
        ProtoFusedSpec::BinScalar(input, op) => {
            array(&[string("bin_scalar"), numeric(*input), string(bin_op_name(*op))])
        }
        ProtoFusedSpec::LeakyRelu(input) => array(&[string("leaky_relu"), numeric(*input)]),
        ProtoFusedSpec::BinPerRow(input, op, mapping) => array(&[
            string("bin_per_row"),
            numeric(*input),
            string(bin_op_name(*op)),
            serialize_mapping(mapping),
        ]),
        ProtoFusedSpec::BinPerCol(input, op, mapping) => array(&[
            string("bin_per_col"),
            numeric(*input),
            string(bin_op_name(*op)),
            serialize_mapping(mapping),
        ]),
        ProtoFusedSpec::AddRowColProducts(row, col) => {
            array(&[string("add_row_col_products"), numeric(*row), numeric(*col)])
        }
        ProtoFusedSpec::AddUnicast(store, input, mapping) => array(&[
            string("add_unicast"),
            numeric(*input),
            serialize_output_store_spec(*store)?,
            serialize_mapping(mapping),
        ]),
        ProtoFusedSpec::Store(store) => {
            array(&[string("store"), serialize_output_store_spec(*store)?])
        }
        ProtoFusedSpec::AddMatMul(geometry, a, b) => array(&[
            string("add_matmul"),
            numeric(*a),
            numeric(*b),
            serialize_add_matmul_geometry(geometry)?,
        ]),
        ProtoFusedSpec::Scaler(_) => {
            bail!("Scaler serialization not implemented — confirm real Scaler{{mult,shift}} fields first")
        }
    })
}

fn bin_op_name(op: BinOp) -> &'static str {
    match op {
        BinOp::Min => "min",
        BinOp::Max => "max",
        BinOp::Add => "add",
        BinOp::Mul => "mul",
        BinOp::Sub => "sub",
        BinOp::SubF => "sub_f",
    }
}

fn rounding_policy_name(policy: RoundingPolicy) -> &'static str {
    match policy {
        RoundingPolicy::Native => "native",
        RoundingPolicy::Zero => "zero",
        RoundingPolicy::Away => "away",
        RoundingPolicy::MinusInf => "minus_inf",
        RoundingPolicy::PlusInf => "plus_inf",
        RoundingPolicy::Even => "even",
        RoundingPolicy::Odd => "odd",
    }
}

fn serialize_mapping(mapping: &MapOutputAxisToInput) -> RValue {
    array(&mapping.0.iter().map(|(a, b)| array(&[numeric(*a), numeric(*b)])).collect::<Vec<_>>())
}

fn serialize_output_store_spec(spec: OutputStoreSpec) -> TractResult<RValue> {
    Ok(match spec {
        OutputStoreSpec::View { m_axis, n_axis, mr, nr } => {
            array(&[string("view"), numeric(m_axis), numeric(n_axis), numeric(mr), numeric(nr)])
        }
        OutputStoreSpec::Strides { row_byte_stride, col_byte_stride, mr, nr } => array(&[
            string("strides"),
            numeric(row_byte_stride),
            numeric(col_byte_stride),
            numeric(mr),
            numeric(nr),
        ]),
    })
}

fn serialize_add_matmul_geometry(geo: &AddMatMulGeometry) -> TractResult<RValue> {
    Ok(array(&[
        serialize_tdim(&geo.k),
        serialize_opt_input_store_spec(&geo.a_storage)?,
        serialize_opt_input_store_spec(&geo.b_storage)?,
        serialize_mapping(&geo.c_to_a_axis_mapping),
        serialize_mapping(&geo.c_to_b_axis_mapping),
    ]))
}

fn serialize_tdim(dim: &TDim) -> RValue {
    match dim {
        TDim::Val(v) => array(&[
            string("val"),
            numeric(*v),
        ]),

        TDim::Sym(s) => array(&[
            string("sym"),
            string(s.to_string()),
        ]),

        TDim::Add(terms) => array(&[
            string("add"),
            array(&terms.iter().map(serialize_tdim).collect::<Vec<_>>()),
        ]),

        TDim::Mul(terms) => array(&[
            string("mul"),
            array(&terms.iter().map(serialize_tdim).collect::<Vec<_>>()),
        ]),

        TDim::MulInt(k, term) => array(&[
            string("mul_int"),
            numeric(*k),
            serialize_tdim(term),
        ]),

        TDim::Div(term, divisor) => array(&[
            string("div"),
            serialize_tdim(term),
            numeric(*divisor),
        ]),
    }
}

fn parse_tdim_value(
    symbol_table: &SymbolTable,
    v: &Value,
) -> TractResult<TDim> {
    let items = as_array(v)?;

    if items.is_empty() {
        bail!("Empty TDim encoding");
    }

    match as_tag(&items[0])? {
        "val" => Ok(TDim::Val(as_i64(&items[1])?)),

        "sym" => {
            let name = as_string(&items[1])?;
            Ok(TDim::Sym(symbol_table.sym(&name)))
        }

        "add" => {
            let terms = as_array(&items[1])?
                .iter()
                .map(|v| parse_tdim_value(symbol_table, v))
                .collect::<TractResult<Vec<_>>>()?;

            Ok(TDim::Add(terms))
        }

        "mul" => {
            let terms = as_array(&items[1])?
                .iter()
                .map(|v| parse_tdim_value(symbol_table, v))
                .collect::<TractResult<Vec<_>>>()?;

            Ok(TDim::Mul(terms))
        }

        "mul_int" => {
            let k = as_i64(&items[1])?;
            let term = parse_tdim_value(symbol_table, &items[2])?;

            Ok(TDim::MulInt(k, Box::new(term)))
        }

        "div" => {
            let term = parse_tdim_value(symbol_table, &items[1])?;
            let divisor = as_i64(&items[2])? as u64;

            Ok(TDim::Div(Box::new(term), divisor))
        }

        tag => bail!("Unknown TDim tag: {}", tag),
    }
}

fn serialize_opt_input_store_spec(spec: &Option<Box<dyn InputStoreSpec>>) -> TractResult<RValue> {
    match spec {
        None => Ok(string("none")),

        Some(spec) => {
            if let Some(prepacked) = spec.as_prepacked() {
                Ok(array(&[
                    string("prepacked"),
                    numeric(prepacked.panel_bytes),
                ]))
            } else {
                bail!("Unsupported InputStoreSpec: {:?}", spec)
            }
        }
    }
}

fn de_lir_matmul_unary(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let inputs: TVec<OutletId> = invocation.named_arg_as(builder, "inputs")?;
    let c_m_axis: usize = invocation.named_arg_as(builder, "c_m_axis")?;
    let c_n_axis: usize = invocation.named_arg_as(builder, "c_n_axis")?;
    let kernel: String = invocation.named_arg_as(builder, "kernel")?;
    let c_datum_type: DatumType = invocation.named_arg_as::<String>(builder, "c_datum_type")?.parse()?;
    let c_shape_value: TVec<Value>  = invocation.named_arg_as(builder, "c_shape")?;
    let c_shape: TVec<TDim> = c_shape_value
        .iter()
        .map(|v| parse_tdim_value(&builder.model.symbol_table, v))
        .collect::<TractResult<_>>()?;
    let c_fact = c_datum_type.fact(&c_shape);

    let micro_ops_value: TVec<Value> = invocation.named_arg_as(builder, "micro_ops")?;
    let micro_ops = micro_ops_value
        .iter()
        .map(|v| parse_proto_fused_spec(&builder.model.symbol_table, v, &kernel))
        .collect::<TractResult<Vec<_>>>()?;

    validate_micro_op_inputs(&micro_ops, inputs.len())?;
    let mmm = kernel_from_name(&kernel)?;

    let op = LirMatMulUnary::new(mmm, c_fact, c_m_axis, c_n_axis, micro_ops)?;
    builder.wire(op, &inputs)
}

fn validate_micro_op_inputs(
    micro_ops: &[ProtoFusedSpec],
    input_count: usize,
) -> TractResult<()> {
    for (ix, spec) in micro_ops.iter().enumerate() {
        let max_input = match spec {
            ProtoFusedSpec::BinScalar(v, _) => Some(*v),
            ProtoFusedSpec::LeakyRelu(v) => Some(*v),
            ProtoFusedSpec::BinPerRow(v, _, _) => Some(*v),
            ProtoFusedSpec::BinPerCol(v, _, _) => Some(*v),
            ProtoFusedSpec::AddRowColProducts(row, col) => Some((*row).max(*col)),
            ProtoFusedSpec::AddUnicast(_, v, _) => Some(*v),
            ProtoFusedSpec::AddMatMul(_, a, b) => Some((*a).max(*b)),
            ProtoFusedSpec::Scaler(_) | ProtoFusedSpec::Store(_) => None,
        };

        if let Some(max_input) = max_input {
            if max_input >= input_count {
                bail!(
                    "LIR micro-op {} references input {}, but LIR has only {} inputs: {:?}",
                    ix,
                    max_input,
                    input_count,
                    spec,
                );
            }
        }
    }

    Ok(())
}

fn as_array(v: &Value) -> TractResult<&[Value]> {
    match v {
        Value::Array(items) => Ok(items),
        Value::Tuple(items) => Ok(items),
        other => bail!("Expected array/tuple value, got {:?}", other),
    }
}

fn as_tag<'v>(v: &'v Value) -> TractResult<&'v str> {
    match v {
        Value::String(s) => Ok(s.as_str()),
        other => bail!("Expected string tag, got {:?}", other),
    }
}

fn as_usize(v: &Value) -> TractResult<usize> {
    match v {
        Value::Scalar(f) => Ok(*f as usize),
        Value::Dim(d) => Ok(d.to_usize()?),
        other => bail!("Expected numeric value, got {:?}", other),
    }
}

fn as_isize(v: &Value) -> TractResult<isize> {
    Ok(as_usize(v)? as isize)
}

fn as_i64(v: &Value) -> TractResult<i64> {
    match v {
        Value::Scalar(f) => Ok(*f as i64),
        Value::Dim(d) => d.to_i64(),
        other => bail!("Expected numeric value, got {:?}", other),
    }
}

fn as_string(v: &Value) -> TractResult<String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        other => bail!("Expected string value, got {:?}", other),
    }
}

fn parse_mapping(v: &Value) -> TractResult<MapOutputAxisToInput> {
    let pairs = as_array(v)?
        .iter()
        .map(|p| {
            let pair = as_array(p)?;
            Ok((as_usize(&pair[0])?, as_usize(&pair[1])?))
        })
        .collect::<TractResult<_>>()?;
    Ok(MapOutputAxisToInput(pairs))
}

fn parse_output_store_spec(v: &Value) -> TractResult<OutputStoreSpec> {
    let items = as_array(v)?;
    match as_tag(&items[0])? {
        "view" => Ok(OutputStoreSpec::View {
            m_axis: as_usize(&items[1])?,
            n_axis: as_usize(&items[2])?,
            mr: as_usize(&items[3])?,
            nr: as_usize(&items[4])?,
        }),
        "strides" => Ok(OutputStoreSpec::Strides {
            row_byte_stride: as_isize(&items[1])?,
            col_byte_stride: as_isize(&items[2])?,
            mr: as_usize(&items[3])?,
            nr: as_usize(&items[4])?,
        }),
        other => bail!("Unknown OutputStoreSpec tag: {}", other),
    }
}

fn parse_input_store_spec(v: &Value) -> TractResult<Option<Box<dyn InputStoreSpec>>> {
    match v {
        Value::String(s) if s == "none" => Ok(None),
        _ => {
            let items = as_array(v)?;
            match as_tag(&items[0])? {
                "prepacked" => {
                    Ok(Some(Box::new(PrepackedSpec { panel_bytes: as_usize(&items[1])? })))
                }
                other => bail!("Unknown InputStoreSpec tag: {}", other),
            }
        }
    }
}

fn parse_add_matmul_geometry(symbol_table: &SymbolTable, v: &Value, kernel: &str) -> TractResult<AddMatMulGeometry> {
    let items = as_array(v)?;
    Ok(AddMatMulGeometry {
        k: parse_tdim_value(symbol_table, &items[0])?,
        a_storage: parse_input_store_spec(&items[1])?,
        b_storage: parse_input_store_spec(&items[2])?,
        mmm: kernel_from_name(kernel)?,
        c_to_a_axis_mapping: parse_mapping(&items[3])?,
        c_to_b_axis_mapping: parse_mapping(&items[4])?,
    })
}

fn parse_proto_fused_spec(symbol_table: &SymbolTable, v: &Value, kernel: &str) -> TractResult<ProtoFusedSpec> {
    let items = as_array(v)?;
    match as_tag(&items[0])? {
        "bin_scalar" => Ok(ProtoFusedSpec::BinScalar(
            as_usize(&items[1])?,
            bin_op_from_name(&as_string(&items[2])?)?,
        )),
        "leaky_relu" => Ok(ProtoFusedSpec::LeakyRelu(as_usize(&items[1])?)),
        "bin_per_row" => Ok(ProtoFusedSpec::BinPerRow(
            as_usize(&items[1])?,
            bin_op_from_name(&as_string(&items[2])?)?,
            parse_mapping(&items[3])?,
        )),
        "bin_per_col" => Ok(ProtoFusedSpec::BinPerCol(
            as_usize(&items[1])?,
            bin_op_from_name(&as_string(&items[2])?)?,
            parse_mapping(&items[3])?,
        )),
        "add_row_col_products" => {
            Ok(ProtoFusedSpec::AddRowColProducts(as_usize(&items[1])?, as_usize(&items[2])?))
        }
        "add_unicast" => Ok(ProtoFusedSpec::AddUnicast(
            parse_output_store_spec(&items[2])?,
            as_usize(&items[1])?,
            parse_mapping(&items[3])?,
        )),
        "store" => Ok(ProtoFusedSpec::Store(parse_output_store_spec(&items[1])?)),
        "add_matmul" => Ok(ProtoFusedSpec::AddMatMul(
            parse_add_matmul_geometry(symbol_table, &items[3], kernel)?,
            as_usize(&items[1])?,
            as_usize(&items[2])?,
        )),
        other => bail!("Unknown ProtoFusedSpec tag: {}", other),
    }
}

fn bin_op_from_name(name: &str) -> TractResult<BinOp> {
    Ok(match name {
        "min" => BinOp::Min,
        "max" => BinOp::Max,
        "add" => BinOp::Add,
        "mul" => BinOp::Mul,
        "sub" => BinOp::Sub,
        "sub_f" => BinOp::SubF,
        other => bail!("Unknown BinOp tag: {}", other),
    })
}

fn kernel_from_name(name: &str) -> TractResult<Box<dyn tract_linalg::frame::mmm::MatMatMul>> {
    match name {
        "avx2_mmm_i32_8x8" => Ok(tract_linalg::x86_64_fma::mmm::avx2_mmm_i32_8x8::mmm()),
        "fma_mmm_f32_8x8" => Ok(tract_linalg::x86_64_fma::mmm::fma_mmm_f32_8x8::mmm()),
        "fma_mmm_f32_16x6" => Ok(tract_linalg::x86_64_fma::mmm::fma_mmm_f32_16x6::mmm()),
        "fma_mmm_f32_24x4" => Ok(tract_linalg::x86_64_fma::mmm::fma_mmm_f32_24x4::mmm()),
        _ => bail!("Unsupported LirMatMulUnary kernel: {}", name),
    }
}