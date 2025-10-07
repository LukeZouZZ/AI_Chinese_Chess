import pickle
from config import CONFIG
import redis

def get_redis_cli():
    """Create a Redis client using CONFIG settings (host/port/db)."""
    r = redis.StrictRedis(host=CONFIG['redis_host'], port=CONFIG['redis_port'], db=CONFIG['redis_db'])
    return r

def get_list_range(redis_cli, name, l, r=-1):
    """Fetch a range of list elements and unpickle them.

    Args:
        redis_cli: an instance of redis.Redis
        name: list key
        l: start index
        r: end index (inclusive if -1)

    Returns:
        A Python list with deserialized objects.
    """
    assert isinstance(redis_cli, redis.Redis)
    list = redis_cli.lrange(name, l, r)  # raw bytes list
    return [pickle.loads(d) for d in list]

if __name__ == '__main__':
    # Example: push training data buffer entries to Redis for consumers
    r = get_redis_cli()
    with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict):
        data_file = pickle.load(data_dict)
        data_buffer = data_file['data_buffer']
    for d in data_buffer:
        r.rpush('train_data_buffer', pickle.dumps(d))
    # Example usage:
    # r.rpush('test', pickle.dumps(([8,2], [2,4], 5)))
    # p = get_list_range(r, 'test', 0, -1)
    # print(p)
