"""Accept SMTP on localhost and save each message as a file."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

HOST = "127.0.0.1"
PORT = 2525
INBOX = Path(__file__).resolve().parents[1] / "runtime" / "mta" / "inbox"


def reply(writer, text):
    writer.write(f"{text}\r\n".encode())


async def handle(reader, writer):
    peer = writer.get_extra_info("peername")
    if not peer or peer[0] not in {"127.0.0.1", "::1"}:
        reply(writer, "550 connections are accepted only from localhost")
        await writer.drain()
        writer.close()
        return
    reply(writer, "220 localhost ESMTP ready")
    await writer.drain()
    sender = ""
    recipients = []
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            command, _, argument = text.partition(" ")
            command = command.upper()
            if command in {"EHLO", "HELO"}:
                reply(writer, "250 localhost")
            elif command == "MAIL":
                sender = argument.split(":", 1)[-1].strip("<> ")
                recipients = []
                reply(writer, "250 OK")
            elif command == "RCPT":
                recipients.append(argument.split(":", 1)[-1].strip("<> "))
                reply(writer, "250 OK")
            elif command == "DATA":
                reply(writer, "354 End with <CRLF>.<CRLF>")
                await writer.drain()
                lines = []
                while True:
                    data = await reader.readline()
                    if data in {b".\r\n", b".\n"}:
                        break
                    if data.startswith(b".."):
                        data = data[1:]
                    lines.append(data)
                INBOX.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                path = INBOX / f"{stamp}.eml"
                path.write_bytes(b"".join(lines))
                print(f"accepted {path.name} from {sender or '?'} to {', '.join(recipients) or '?'}", flush=True)
                sender = ""
                recipients = []
                reply(writer, "250 OK")
            elif command == "RSET":
                sender = ""
                recipients = []
                reply(writer, "250 OK")
            elif command == "NOOP":
                reply(writer, "250 OK")
            elif command == "QUIT":
                reply(writer, "221 Bye")
                await writer.drain()
                break
            else:
                reply(writer, "502 command not implemented")
            await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def main():
    server = await asyncio.start_server(handle, HOST, PORT)
    print(f"SMTP listening on {HOST}:{PORT}", flush=True)
    print(f"Messages are saved in {INBOX}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
