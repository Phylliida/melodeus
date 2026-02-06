from dataclasses import dataclass
from dataclasses import asdict
import os
import asyncio
from pathlib import Path
import traceback
from async_callback_manager import AsyncCallbackManager
from enum import Enum
import json

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

class AsyncContextManager(object):
    def __init__(self, config, voices):
        self.messages = []
        self.history = []
        self.config = config
        self.voices = voices
        self.context_update_callback = AsyncCallbackManager()
    
    async def __aenter__(self):
        await self.load_context()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def add_callback(self, callback):
        await self.context_update_callback.add_callback(callback)

    async def remove_callback(self, callback):
        await self.context_update_callback.remove_callback(callback)

    async def apply_context_update(self, update, record=False):
        if update.action == ContextAction.CREATE or update.action == ContextAction.EDIT:
            await self.update(update.uuid, update.author, update.message, record=record)
        elif update.action == ContextAction.DELETE:
            await self.delete(update.uuid, record=record)

    async def load_context(self):
        if os.path.exists(self.config.history):
            with open(self.config.history, "r", encoding="utf-8") as f:
                for line in f.read().split("\n"):
                    if line.strip() == "": continue
                    try:
                        update = ContextUpdate(**json.loads(line))
                        await self.apply_context_update(update, record=False)
                    except json.JSONDecodeError as e:
                        # handle or log the error
                        print(f"Failed to parse JSON")
                        print(line)
                        print(traceback.print_exc())
                    

    def persist(self, context_update):
        dest = Path(self.config.history)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(context_update)) + "\n")

    def encode_context(self):
        res = []
        for message in self.messages:
            res.append(message.author + ": " + message.message)
        return "\n".join(res)

    async def fetch_current_state(self, callback):
        for message in self.messages:
            update = ContextUpdate(
                action=ContextAction.CREATE,
                uuid=message.uuid,
                author=message.author,
                message=message.message
            )
            if asyncio.iscoroutinefunction(callback):
                await callback(update)
            else:
                callback(update)

    async def emit(self, context_update, record=True):
        await self.context_update_callback(context_update)
        self.history.append(context_update)
        if record:
            self.persist(context_update)

    async def update(self, uuid, author, message, record=True):
        matches = [x for x in self.messages if x.uuid == uuid]
        # edit existing message
        if len(matches) > 0:
            matches[0].author = author
            matches[0].message = message
            await self.emit(ContextUpdate(
                action=ContextAction.EDIT,
                uuid=uuid,
                author=author,
                message=message), record=record)
        else:
            self.messages.append(ContextMessage(
                uuid=uuid,
                author=author,
                message=message
            ))
            await self.emit(ContextUpdate(
                action=ContextAction.CREATE,
                uuid=uuid,
                author=author,
                message=message
            ), record=record)

    async def delete(self, uuid, record=True):
        matches = [(i, x) for (i,x) in enumerate(self.messages) if x.uuid == uuid]
        # edit existing message
        if len(matches) > 0:
            match_i, match = matches[0]
            self.messages.pop(match_i)
            await self.emit(ContextUpdate(
                action=ContextAction.DELETE,
                uuid=uuid,
                author="",
                message=""
            ), record=record)


    