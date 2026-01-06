import traceback
import asyncio

ADD_CALLBACK = "add"
REMOVE_CALLBACK = "remove"

class AsyncCallbackManager(object):
    def __init__(self):
        self.callback_queue = asyncio.Queue(maxsize=0)
        self.callbacks = []
    
    async def add_callback(self, callback):
        await self.callback_queue.put((ADD_CALLBACK, callback))

    async def remove_callback(self, callback):
        await self.callback_queue.put((REMOVE_CALLBACK, callback))

    async def __call__(self, *args, **kwargs):
        while not self.callback_queue.empty():
            callback_message_type, callback = self.callback_queue.get_nowait()
            self.callback_queue.task_done() # weird stuff callback queue wants
            if callback_message_type is not None:
                match callback_message_type:
                    case "add":
                        self.callbacks.append(callback)
                    case "remove":
                        if callback in self.callbacks:
                            self.callbacks.remove(callback)
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except:
                print("Error in callback")
                print(traceback.print_exc())


