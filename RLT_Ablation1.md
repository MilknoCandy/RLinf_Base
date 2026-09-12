  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
    raise self._exception
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xx/RLinf_Base/rlinf/scheduler/worker/worker.py", line 92, in async_func
    return await func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xx/RLinf_Base/rlinf/scheduler/worker/worker.py", line 1367, in wrapper
    return await func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xx/RLinf_Base/rlinf/workers/rollout/hf/async_huggingface_worker.py", line 60, in generate
    await self._generate_task
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xx/RLinf_Base/rlinf/workers/rollout/hf/async_huggingface_worker.py", line 78, in _generate
    await self.generate_one_epoch(input_channel, output_channel)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xx/RLinf_Base/rlinf/scheduler/worker/worker.py", line 92, in async_func
    return await func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xx/RLinf_Base/rlinf/scheduler/worker/worker.py", line 1367, in wrapper
    return await func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xx/RLinf_Base/rlinf/workers/rollout/hf/huggingface_worker.py", line 730, in generate_one_epoch
    actions, result = self._predict_rollout_actions(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xx/RLinf_Base/rlinf/workers/rollout/hf/huggingface_worker.py", line 598, in _predict_rollout_actions
    return predict_rlt_actions(
           ^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xx/RLinf_Base/rlinf/algorithms/rlt/rollout.py", line 63, in predict_rlt_actions
    rlt_obs = feature_model.extract_rlt_obs(env_obs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xx/RLinf_Base/rlinf/models/embodiment/openpi/openpi_action_model.py", line 569, in extract_rlt_obs
    self._build_rlt_prefix_cache(observation, train=False)
  File "/xxx/xx/RLinf_Base/rlinf/models/embodiment/openpi/openpi_action_model.py", line 509, in _build_rlt_prefix_cache
    prefix_output, prefix_pad_masks, past_key_values = self._build_prefix_cache(
                                                       ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xx/RLinf_Base/rlinf/models/embodiment/openpi/openpi_action_model.py", line 1259, in _build_prefix_cache
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                                                      ^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/openpi/models_pytorch/pi0_pytorch.py", line 202, in embed_prefix
    img_emb = self._apply_checkpoint(image_embed_func, img)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/openpi/models_pytorch/pi0_pytorch.py", line 154, in _apply_checkpoint
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/openpi/models_pytorch/pi0_pytorch.py", line 200, in image_embed_func
    return self.paligemma_with_expert.embed_image(img)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/openpi/models_pytorch/gemma_pytorch.py", line 86, in embed_image
    return self.paligemma.model.get_image_features(image)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/transformers/models/paligemma/modeling_paligemma.py", line 242, in get_image_features
    image_outputs = self.vision_tower(pixel_values)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/transformers/utils/generic.py", line 943, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/transformers/models/siglip/modeling_siglip.py", line 870, in forward
    return self.vision_model(
           ^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/transformers/utils/generic.py", line 943, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/transformers/models/siglip/modeling_siglip.py", line 777, in forward
    encoder_outputs: BaseModelOutput = self.encoder(
                                       ^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/transformers/utils/generic.py", line 943, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/transformers/models/siglip/modeling_siglip.py", line 608, in forward
    layer_outputs = encoder_layer(
                    ^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/transformers/modeling_layers.py", line 83, in __call__
    return super().__call__(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/transformers/models/siglip/modeling_siglip.py", line 462, in forward
    hidden_states = self.layer_norm1(hidden_states)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/torch/nn/modules/normalization.py", line 217, in forward
    return F.layer_norm(
           ^^^^^^^^^^^^^
  File "/xxx/xxx/xxx/xxx/miniconda3/envs/sdw_rlinf-pi312/lib/python3.12/site-packages/torch/nn/functional.py", line 2910, in layer_norm
    return torch.layer_norm(
           ^^^^^^^^^^^^^^^^^
RuntimeError: expected scalar type Float but found BFloat16
Exception occurred while running AsyncMultiStepRolloutWorker's function sync_model_from_actor: exception is The actor died unexpectedly before finishing this task.
