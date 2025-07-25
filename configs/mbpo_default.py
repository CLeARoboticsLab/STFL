import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = 'mbpo'

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.tau = 0.005

    config.init_temperature = 1.0
    config.target_entropy = None

    return config
