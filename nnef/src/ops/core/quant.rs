use crate::internal::*;
use tract_core::ops::quant::DequantizeLinearF32;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_dequantize_linear_f32);
    
    registry.register_primitive(
        "tract_core_dequantize_linear_f32",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("scale"),
            TypeName::Integer.named("zero_point"),
        ],
        &[("output", TypeName::Integer.tensor())],
        de_dequantize_linear_f32,
    );
}

fn ser_dequantize_linear_f32(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &DequantizeLinearF32,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();

    Ok(Some(invocation(
        "tract_core_dequantize_linear_f32",
        &[input],
        &[
            ("scale", numeric(op.scale)),
            ("zero_point", numeric(op.zero_point)),
        ],
    )))
}

fn de_dequantize_linear_f32(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let scale: f32 = invocation.named_arg_as(builder, "scale")?;
    let zero_point: i64 = invocation.named_arg_as(builder, "zero_point")?;

    builder.wire(
        DequantizeLinearF32::new(scale, zero_point as i32),
        &[input],
    )
}