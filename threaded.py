# see https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread-in-python/14331755#14331755

def threaded(f, daemon=False):
    import threading
    import Queue

    def wrapper(q, *args, **kwargs):
        '''this function calls the decorated function and puts the result in a queue'''
        ret = f(*args, **kwargs)
        q.put(ret)

    def wrap(*args, **kwargs):
        '''this is the function returned from the decorator. It fires off wrapper 
        in a new thread and returns the thread object with the result queue attached'''
        q = Queue.Queue()
        t = threading.Thread(target=wrapper, args = (q,) + args, kwargs = kwargs)
        t.daemon = daemon
        t.result = q        
        return t

    return wrap
