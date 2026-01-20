from dataclasses import dataclass
from async_callback_manager import async_callback_manager
from enum import Enum

@dataclass
class ContextMessage:
    uuid: str
    author: str
    message: str

class ContextAction(str, Enum):
    CREATE = "create"
    EDIT = "edit"
    DELETE = "delete"

@dataclass
class ContextUpdate:
    action: ContextAction
    uuid: str
    author: str
    message: str

class ContextManager(object):
    def __init__(self):
        self.messages = []
        self.context_update_callback = AsyncCallbackManager()  

    async def add_callback(self, callback):
        await self.context_update_callback.add_callback(callback)

    async def remove_callback(self, callback):
        await self.context_update_callback.remove_callback(callback)

    async def emit(self, context_update):
        await self.context_update_callback(context_update)

    async def update(self, uuid, author, message):
        matches = [x for x in self.messages if x.uuid == uuid]
        # edit existing message
        if len(matches) > 0:
            matches[0].author = author
            matches[0].message = message
            await self.emit(ContextUpdate(
                action=ContextAction.EDIT
                uuid=uuid,
                author=author,
                message=message))
        else:
            messages.append(ContextMessage(
                uuid=uuid,
                author=author,
                message=message
            ))
            await self.emit(ContextUpdate(
                action=ContextAction.CREATE,
                uuid=uuid,
                author=author,
                message=message
            ))

    async def delete(self, uuid):
        matches = [(i, x) for (i,x) in enumerate(self.messages) if x.uuid == uuid]
        # edit existing message
        if len(matches) > 0:
            match_i, match = matches[0]
            self.messages.pop(match_i)
            await self.emit(ContextUpdate(
                action=ContextAction.DELETE,
                uuid=uuid,
                author="",
                context=""
            ))


    