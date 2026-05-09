"""Simple CLI to manage automation tasks.

Usage:
    python scripts/automation_cli.py schedule --user me --type send_reminder --delay 60 --message "Hello"
    python scripts/automation_cli.py list --user me
    python scripts/automation_cli.py cancel --id 2
"""
import argparse
from datetime import datetime, timedelta
import json
import config as _config
from automation.automation_engine import AutomationEngine

ENGINE = AutomationEngine(str(_config.BASE_DIR / "automation.db"))


def cmd_schedule(args):
    payload = {}
    if args.message:
        payload["message"] = args.message
    scheduled = datetime.utcnow() + timedelta(seconds=args.delay) if args.delay else datetime.utcnow()
    tid = ENGINE.schedule_task(user_id=args.user, task_type=args.type, task_payload=payload, scheduled_time=scheduled, recurrence=args.recurrence)
    print("scheduled", tid)


def cmd_list(args):
    tasks = ENGINE.list_tasks(args.user)
    print(json.dumps(tasks, default=str, indent=2))


def cmd_cancel(args):
    ENGINE.cancel_task(int(args.id))
    print("cancelled", args.id)


parser = argparse.ArgumentParser()
sub = parser.add_subparsers()

p = sub.add_parser("schedule")
p.add_argument("--user", default="me")
p.add_argument("--type", required=True)
p.add_argument("--delay", type=int, default=0)
p.add_argument("--message")
p.add_argument("--recurrence")
p.set_defaults(func=cmd_schedule)

p2 = sub.add_parser("list")
p2.add_argument("--user", default=None)
p2.set_defaults(func=cmd_list)

p3 = sub.add_parser("cancel")
p3.add_argument("--id", required=True)
p3.set_defaults(func=cmd_cancel)

if __name__ == "__main__":
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
    else:
        args.func(args)
