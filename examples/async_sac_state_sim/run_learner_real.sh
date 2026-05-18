export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_state_real.py "$@" \
    --learner \
    --env Auboi5_assemble_hole_env-v0 \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --training_starts 20 \
    --critic_actor_ratio 8 \
    --batch_size 128 \
    --replay_buffer_capacity 100000\
    --loaded_model ~/serl/examples/async_sac_state_sim/model_savings/ \
    --debug # wandb is disabled when debug
    # --checkpoint_period 2000000 \
    # --checkpoint_path ~/serl/examples/async_sac_state_sim/model_savings \
