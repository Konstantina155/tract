use crate::internal::*;
use tract_core::ops::matmul::pack::MatMatMulPack;
use tract_linalg::frame::Packer;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_mat_matmul_pack);

    registry.register_primitive(
        "tract_core_matmul_pack",
        &pack_parameters(),
        &[("output", TypeName::Scalar.tensor())],
        de_mat_matmul_pack,
    );
}

fn pack_parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Integer.named("r"),
        TypeName::Integer.named("alignment"),
        TypeName::Integer.named("end_padding_record"),
        TypeName::Integer.named("k_axis"),
        TypeName::Integer.named("mn_axis"),
    ]
}

fn ser_mat_matmul_pack(ast: &mut IntoAst, node: &TypedNode, op: &MatMatMulPack) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();

    let packer = op.packer();
    let k_axis = op.k_axis();
    let mn_axis = op.mn_axis();

    Ok(Some(invocation(
        "tract_core_matmul_pack",
        &[input],
        &[
            ("r", numeric(packer.r)),
            ("alignment", numeric(packer.alignment())),
            ("end_padding_record", numeric(packer.end_padding_record())),
            ("k_axis", numeric(k_axis)),
            ("mn_axis", numeric(mn_axis)),
        ],
    )))
}

fn de_mat_matmul_pack(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input: OutletId = invocation.named_arg_as(builder, "input")?;
    let r: usize = invocation.named_arg_as(builder, "r")?;
    let alignment: usize = invocation.named_arg_as(builder, "alignment")?;
    let end_padding_record: usize = invocation.named_arg_as(builder, "end_padding_record")?;
    let k_axis: usize = invocation.named_arg_as(builder, "k_axis")?;
    let mn_axis: usize = invocation.named_arg_as(builder, "mn_axis")?;

    let input_shape = builder.model.outlet_fact(input)?.shape.clone();
    let packer = Packer::new(r, alignment, end_padding_record);
    let op = MatMatMulPack::new(packer, k_axis, mn_axis, &input_shape);

    builder.wire(op, &[input])
}