export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_state_real.py "$@" \
    --actor \
    --render \
    --env Auboi5_assemble_hole_env-v0 \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --random_steps 20 \
    --debug \
    --loaded_model ~/serl/examples/async_sac_state_sim/model_savings/ \
    # --record_trajectory \
