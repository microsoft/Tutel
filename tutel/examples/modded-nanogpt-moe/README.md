This example is modified based on https://github.com/KellerJordan/modded-nanogpt by replacing the Dense layer with MoE layer.

Steps to Train:

(1) Define moe_layer in transformers with expected weight dtype (e.g. torch.bfloat16). [(Here to follow)](train_gpt_v0.py#L199-L204)

(2) Broadcast shared parameter weights to all other GPUs while preventing broadcasting of non-shared parameters. [(Here to follow)](train_gpt_v0.py#L378-L379)

(3) Perform a single step training and perform an all-reduce on the gradients of all shared parameters while preventing all-reduce on non-shared parameters. [(Here to follow)](train_gpt_v0.py#L523)

(4) Run `opt.step()` to apply the distributed gradients [(Here to follow)](train_gpt_v0.py#L527-L528)


```sh
# run MoE training with 8 local GPUs
USE_MOE=1 ./run.sh

# run Dense training with 8 local GPUs
USE_MOE=0 ./run.sh
```

```txt
...
[DEBUG(weights-init)] Validating all parameters are in proper states..
[DEBUG(weights-init)] Validation passed. ✅
[DEBUG(gradients)] Validating all parameters are in proper states..
[DEBUG(gradients)] Validation passed. ✅
[DEBUG(weights-upd)] Validating all parameters are in proper states..
[DEBUG(weights-upd)] Validation passed. ✅
step:50/1750 val_loss:7.6209 val_balance_loss:8.5625 step_time:153.01ms
step:100/1750 val_loss:7.1094 val_balance_loss:7.7812 step_time:137.49ms
step:150/1750 val_loss:6.7864 val_balance_loss:7.7188 step_time:138.65ms
step:200/1750 val_loss:6.6302 val_balance_loss:7.5625 step_time:139.24ms
step:250/1750 val_loss:6.4994 val_balance_loss:6.7812 step_time:139.54ms
step:300/1750 val_loss:6.4303 val_balance_loss:6.7500 step_time:139.80ms
step:350/1750 val_loss:6.3354 val_balance_loss:4.5000 step_time:140.29ms
step:400/1750 val_loss:6.2024 val_balance_loss:4.2812 step_time:140.41ms
step:450/1750 val_loss:6.1186 val_balance_loss:4.2812 step_time:140.39ms
step:500/1750 val_loss:6.0563 val_balance_loss:4.2812 step_time:140.42ms
step:550/1750 val_loss:5.9913 val_balance_loss:4.2812 step_time:144.79ms
step:600/1750 val_loss:5.9157 val_balance_loss:4.2812 step_time:140.01ms
step:650/1750 val_loss:5.8420 val_balance_loss:4.2812 step_time:139.76ms
step:700/1750 val_loss:5.7899 val_balance_loss:3.9219 step_time:139.85ms
...
```