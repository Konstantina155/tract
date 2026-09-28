use tract_nnef::internal::*;
use tract_ndarray::ArrayViewD;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_onnx_dynamic_quantize_linear_u8",
        &[
            TypeName::Scalar.tensor().named("input"),
        ],
        &[("outputs", TypeName::Scalar.tensor().array())],
        load,
    );
    registry.register_dumper(dump);
}

pub fn dump(
    ast: &mut IntoAst,
    node: &TypedNode,
    _op: &DynamicQuantizeLinearU8,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();

    Ok(Some(invocation(
        "tract_onnx_dynamic_quantize_linear_u8",
        &[input],
        &[],
    )))
}

pub fn load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let op: Box<dyn TypedOp> = Box::new(DynamicQuantizeLinearU8);

    builder.wire(
        op,
        &[input],
    )
}

fn dynamic_quantize_linear_f32_u8(x: f32, scale: f32, zero_point: u8) -> u8 {
    (((x / scale).round() as i32) + zero_point as i32)
        .clamp(u8::min_value() as i32, u8::max_value() as i32) as u8
}

fn dynamic_quantize_linear_u8(scale: f32, zero_point: u8, xs: &[f32], ys: &mut [u8]) {
    xs.iter()
        .zip(ys.iter_mut())
        .for_each(|(x, y)| {
            *y = dynamic_quantize_linear_f32_u8(*x, scale, zero_point)
        });
}

fn scale_and_zero_point(v: ArrayViewD<f32>) -> (f32, u8) {
    let (min, max) = v.fold((0., 0.), |(a_min, a_max), &v| {
        if v < a_min {
            (v, a_max)
        } else if v > a_max {
            (a_min, v)
        } else {
            (a_min, a_max)
        }
    });

    let min_t = u8::min_value() as f32;
    let max_t = u8::max_value() as f32;

    let scale = (max - min) / max_t;

    let zero_point = -min / scale;
    let zero_point = zero_point.round();
    let zero_point = zero_point.max(min_t);
    let zero_point = zero_point.min(max_t);

    (scale, zero_point as u8)
}

#[derive(Clone, Debug, Hash)]
pub struct DynamicQuantizeLinearU8;

impl Op for DynamicQuantizeLinearU8 {
    fn name(&self) -> Cow<str> {
        "DynamicQuantizeLinearU8".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![])
    }

    fn validation(&self) -> Validation {
        Validation::Accurate
    }

    op_as_typed_op!();
}

impl EvalOp for DynamicQuantizeLinearU8 {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = &inputs[0];
        let input = input.cast_to::<f32>()?;
        let a_input = input.to_array_view::<f32>()?;

        let (scale, zero_point) = scale_and_zero_point(a_input);

        let mut dst =
            unsafe { Tensor::uninitialized_dt(u8::datum_type(), input.shape())? };

        dynamic_quantize_linear_u8(
            scale,
            zero_point,
            input.as_slice::<f32>()?,
            dst.as_slice_mut::<u8>()?,
        );

        let quantized_tensor = dst.into_tvalue();
        let scale_tensor = tensor0(scale).into();
        let zero_point_tensor = tensor0(zero_point).into();

        Ok(tvec!(
            quantized_tensor,
            scale_tensor,
            zero_point_tensor
        ))
    }
}

impl TypedOp for DynamicQuantizeLinearU8 {
    fn output_facts(
        &self,
        inputs: &[&TypedFact],
    ) -> TractResult<TVec<TypedFact>> {
        let mut quantized_fact = inputs[0].clone();
        quantized_fact.datum_type = u8::datum_type();

        let scale_fact = f32::fact([0; 0]);
        let zero_fact = u8::fact([0; 0]);

        Ok(tvec!(
            quantized_fact,
            scale_fact,
            zero_fact
        ))
    }

    as_op!();
}