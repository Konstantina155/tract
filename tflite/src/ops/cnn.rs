use super::wire_fused_activation;
use crate::registry::{DeserOp, Registry};
use crate::ser::{BuiltinOp, SubgraphBuilder};
use crate::tflite::{
    ActivationFunctionType, BuiltinOperator, BuiltinOptions, Conv2DOptions, Conv2DOptionsArgs,
    DepthwiseConv2DOptions, DepthwiseConv2DOptionsArgs, PadOptions, PadOptionsArgs, Padding,
};
use tract_core::internal::*;
use tract_core::ops as core;
use tract_core::ops::array::{Pad, PadMode};
use tract_core::ops::cast::cast;
use tract_core::ops::cnn::KernelFormat;
use tract_core::ops::cnn::{ConvUnary, PaddingSpec};
use tract_core::ops::nn::DataFormat;
use tract_core::prelude::tract_itertools::Itertools;

pub fn register_all(reg: &mut Registry) {
    reg.reg_to_tract(BuiltinOperator::AVERAGE_POOL_2D, average_pool_2d);
    reg.reg_to_tract(BuiltinOperator::CONV_2D, de_conv2d);
    reg.reg_to_tflite(ser_conv);
    reg.reg_to_tract(BuiltinOperator::DEPTHWISE_CONV_2D, de_dw_conv2d);
    reg.reg_to_tflite(ser_pad);
}

fn average_pool_2d(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let options = builtin!(op, builtin_options_as_pool_2_doptions);
    let strides = tvec!(options.stride_h() as usize, options.stride_w() as usize);
    let kernel_shape = tvec!(options.filter_height() as usize, options.filter_width() as usize);
    let padding = match options.padding() {
        Padding::SAME => PaddingSpec::SameUpper,
        Padding::VALID => PaddingSpec::Valid,
        _ => todo!(),
    };
    let ci = DataFormat::NHWC
        .shape(&op.facts()?[0].shape)?
        .c()
        .to_usize()
        .context("Except defined integer depth")?;
    let pool_spec = core::cnn::PoolSpec {
        data_format: DataFormat::NHWC,
        kernel_shape,
        padding,
        strides: Some(strides),
        dilations: None,
        input_channels: ci,
        output_channels: ci,
    };
    let pool = core::cnn::SumPool { pool_spec, normalize: true, count_include_pad: false };
    let wires = op.ctx.target.wire_node(op.prefix, pool, &op.inputs[0..1])?;
    wire_fused_activation(op, &wires, &options.fused_activation_function())
}

fn ser_conv(
    builder: &mut SubgraphBuilder,
    model: &TypedModel,
    node: &TypedNode,
    conv: &ConvUnary,
) -> TractResult<()> {
    ensure!(conv.pool_spec.data_format == DataFormat::NHWC);
    ensure!(model.node_input_facts(node.id)?[0].rank() == 4);
    ensure!(conv.kernel_fmt == KernelFormat::OHWI);
    ensure!(conv.group == 1 || conv.group.to_dim() == model.node_input_facts(node.id)?[0].shape[3]);
    ensure!(
        conv.pool_spec.padding == PaddingSpec::Valid
            || conv.pool_spec.padding == PaddingSpec::SameUpper
    );
    let node_name = &node.name;
    let mut inputs = tvec!(builder.outlets_to_tensors[&node.inputs[0]],);
    if conv.q_params.is_some() {
        /*
        let facts = model.node_input_facts(node.id)?;
        let iscale = facts[0].datum_type.zp_scale().1;
        let k0_tract = facts[1].konst.as_ref().unwrap().cast_to_scalar::<i32>()? as i64;
        let kscale = facts[2].konst.as_ref().unwrap().as_slice::<f32>()?;
        let per_channel = !kscale.iter().all_equal();
        if per_channel {
            inputs.push(builder.write_fact_with_per_axis_q(
                &format!("{node_name}.weights"),
                &conv.kernel,
                &vec![k0_tract; conv.kernel.shape()[0]],
                kscale,
                0,
            )?);
            let bscale = kscale.iter().map(|k| k * iscale).collect_vec();
            bias = bias.clone().into_tensor().cast_to::<i32>()?.into_owned().into_arc_tensor();
            inputs.push(builder.write_fact_with_per_axis_q(
                &format!("{node_name}.bias"),
                &bias,
                &vec![0i64; bias.len()],
                &bscale,
                0,
            )?);
        } else {
            inputs.push(builder.write_fact(&format!("{node_name}.weights"), &conv.kernel)?);
            let bias_qdt = bias
                .datum_type()
                .quantize(QParams::ZpScale { zero_point: 0, scale: iscale * kscale[0] });
            let bias = bias.cast_to_dt(bias_qdt)?.into_owned();
            inputs.push(builder.write_fact(&format!("{node_name}.bias"), bias)?);
        }
        */
        todo!();
    } else {
        inputs.push(builder.outlets_to_tensors[&node.inputs[1]]);
        inputs.push(builder.outlets_to_tensors[&node.inputs[2]]);
    }
    let output = builder.outlets_to_tensors[&node.id.into()];

    let padding =
        if conv.pool_spec.padding == PaddingSpec::Valid { Padding::VALID } else { Padding::SAME };
    if conv.group == 1 {
        let options = Conv2DOptions::create(
            builder.fb(),
            &Conv2DOptionsArgs {
                padding,
                stride_h: conv.pool_spec.stride(0) as _,
                stride_w: conv.pool_spec.stride(1) as _,
                dilation_h_factor: conv.pool_spec.dilation(0) as _,
                dilation_w_factor: conv.pool_spec.dilation(1) as _,
                fused_activation_function: ActivationFunctionType::NONE,
            },
        );
        builder.write_op_with_options(
            &inputs,
            &[output],
            BuiltinOp::new(3, 2, BuiltinOperator::CONV_2D, BuiltinOptions::Conv2DOptions),
            options.as_union_value(),
        )
    } else {
        let depth_multiplier = (conv.pool_spec.output_channels / conv.group) as i32;
        let options = DepthwiseConv2DOptions::create(
            builder.fb(),
            &DepthwiseConv2DOptionsArgs {
                padding,
                depth_multiplier,
                stride_h: conv.pool_spec.stride(0) as _,
                stride_w: conv.pool_spec.stride(1) as _,
                dilation_h_factor: conv.pool_spec.dilation(0) as _,
                dilation_w_factor: conv.pool_spec.dilation(1) as _,
                fused_activation_function: ActivationFunctionType::NONE,
            },
        );
        builder.write_op_with_options(
            &inputs,
            &[output],
            BuiltinOp::new(
                4,
                2,
                BuiltinOperator::DEPTHWISE_CONV_2D,
                BuiltinOptions::DepthwiseConv2DOptions,
            ),
            options.as_union_value(),
        )
    }
}

fn de_conv2d(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let (input, kernel, bias) = args_3!(op.facts()?);
    let kernel_full_shape = kernel.shape.as_concrete().context("Expect concrete kernel shape")?;
    let kernel_spatial_shape = KernelFormat::OHWI.spatial_shape(kernel_full_shape);
    let options = builtin!(op, builtin_options_as_conv_2_doptions);
    let padding = match options.padding() {
        Padding::SAME => PaddingSpec::SameUpper,
        Padding::VALID => PaddingSpec::Valid,
        _ => todo!(),
    };
    let strides = tvec!(options.stride_h() as usize, options.stride_w() as usize);
    let dilations =
        tvec!(options.dilation_h_factor() as usize, options.dilation_w_factor() as usize);
    let input_channels = *KernelFormat::OHWI.i(kernel_full_shape);
    let output_channels = *KernelFormat::OHWI.o(kernel_full_shape);
    let pool_spec = core::cnn::PoolSpec {
        data_format: tract_core::ops::nn::DataFormat::NHWC,
        kernel_shape: kernel_spatial_shape.into(),
        padding,
        strides: Some(strides),
        dilations: Some(dilations),
        input_channels,
        output_channels,
    };
    let mut inputs = tvec!(op.inputs[0]);
    let q_params = super::linearops_quantization_suport(op, &input, &mut inputs)?;
    let bias_dt = bias.datum_type.unquantized();
    let bias = op.ctx.target.wire_node(
        format!("{}.cast_bias", op.prefix),
        cast(bias_dt),
        &[op.inputs[2]],
    )?[0];
    let conv =
        core::cnn::ConvUnary { pool_spec, kernel_fmt: KernelFormat::OHWI, group: 1, q_params };
    let wires = op.ctx.target.wire_node(op.prefix, conv, &inputs)?;
    wire_fused_activation(op, &wires, &options.fused_activation_function())
}

fn de_dw_conv2d(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let (input, kernel, bias) = args_3!(op.facts()?);
    let bias = bias.konst.unwrap();
    let kernel = kernel.konst.unwrap();
    let kernel_full_shape: TVec<usize> = kernel.shape().into();
    let kernel_shape: TVec<usize> = KernelFormat::OHWI.spatial_shape(&kernel_full_shape).into();
    let options = builtin!(op, builtin_options_as_depthwise_conv_2_doptions);
    let padding = match options.padding() {
        Padding::SAME => PaddingSpec::SameUpper,
        Padding::VALID => PaddingSpec::Valid,
        _ => todo!(),
    };
    let strides = tvec!(options.stride_h() as usize, options.stride_w() as usize);
    let dilations =
        tvec!(options.dilation_h_factor() as usize, options.dilation_w_factor() as usize);
    let output_channels = *KernelFormat::OHWI.i(&kernel_full_shape);
    let pool_spec = core::cnn::PoolSpec {
        data_format: tract_core::ops::nn::DataFormat::NHWC,
        kernel_shape,
        padding,
        strides: Some(strides),
        dilations: Some(dilations),
        input_channels: output_channels,
        output_channels,
    };
    let mut inputs = tvec!(op.inputs[0]);
    let q_params = super::linearops_quantization_suport(op, &input, &mut inputs)?;
    let conv = core::cnn::ConvUnary {
        pool_spec,
        kernel_fmt: KernelFormat::OHWI,
        group: output_channels,
        q_params,
    };
    let wires = op.ctx.target.wire_node(op.prefix, conv, &inputs)?;
    wire_fused_activation(op, &wires, &options.fused_activation_function())
}

fn ser_pad(
    builder: &mut SubgraphBuilder,
    _model: &TypedModel,
    node: &TypedNode,
    pad: &Pad,
) -> TractResult<()> {
    let node_name = &node.name;
    let mut inputs = tvec!(builder.outlets_to_tensors[&node.inputs[0]]);
    let outputs = (0..node.outputs.len())
        .map(|o| builder.outlets_to_tensors[&OutletId::new(node.id, o)])
        .collect_vec();
    let paddings = tract_ndarray::Array2::<i32>::from_shape_fn((pad.pads.len(), 2), |(d, side)| {
        (if side == 0 { pad.pads[d].0 } else { pad.pads[d].1 }) as i32
    });
    inputs.push(builder.write_fact(format!("{node_name}.paddings"), paddings.into_tensor())?);
    let PadMode::Constant(pad_value) = &pad.mode else {
        bail!("Only constant padding is supported by tflite");
    };
    inputs.push(builder.write_fact(format!("{node_name}.pad_value"), pad_value)?);
    let options = PadOptions::create(builder.fb(), &PadOptionsArgs {});
    builder.write_op_with_options(
        &inputs,
        &outputs,
        BuiltinOp::new(60, 1, BuiltinOperator::PADV2, BuiltinOptions::PadV2Options),
        options.as_union_value(),
    )?;
    Ok(())
}
