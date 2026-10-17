"""Explicit static one/two-page protocol over the existing contained resident Session.

No scheduler or device thread is added. Caller threads rendezvous before one submission; the existing
reader validates page mapping and the aggregate retirement barrier. A canceled client keeps its own
owner and drains events. Only cancellation of every submitted client sends a group cancel, once.
"""
import secrets
import threading

from . import protocol as P
from .backend import Session, _Request, NativeFailure, RECENT, _request_identity


class LockedProcessor:
    """Keep existing processor/tokenizer calls serial while native owners may share decode work."""
    def __init__(self, processor):
        self.processor, self.lock = processor, threading.Lock()

    @property
    def runtime(self):
        return self.processor.runtime

    def process(self, image, text):
        with self.lock:
            return self.processor.process(image, text)

    def decode(self, ids):
        with self.lock:
            return self.processor.decode(ids)

    def unload(self):
        with self.lock:
            return self.processor.unload()


class CohortSession(Session):
    """Dedicated native mode: at most two owners, one stable group, no submitted request replay."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending = []
        self.pending_since = None
        self.group = None
        self.owner_count = None  # Installed by the existing backend; standalone callers keep singleton behavior.

    def submit(self, prepared, cancel, request_id=None):
        request = _Request(_request_identity(request_id), "o-" + secrets.token_hex(16), prepared)
        request.state = "pending"
        request.client_cancel = cancel
        request.cohort_terminal = None
        request.cohort_released = False
        with self.cond:
            if self.ready is None or self.unusable or self.closed:
                raise NativeFailure("Native cohort worker is not accepting requests")
            if len(self.requests) >= 2:
                raise NativeFailure("Two native cohort owners remain held")
            if request.id in self.requests or request.id in self.recent:
                raise NativeFailure("Native request identity is already owned or recently retired")
            self.requests[request.id] = request
            self.pending.append(request)
            if self.pending_since is None:
                self.pending_since = self.clock()
            # Queued owners remain subject to the original HTTP Job deadline/cancel and finite owner lifetime.
            # The native acknowledgement budget begins only at the one actual submit_cohort write.
            deadline = None
            self.cond.notify_all()
            while request.state in ("pending", "submitted") and not self.closed:
                now = self.clock()
                if request.state == "pending" and (cancel.is_set() or self.unusable):
                    self.pending.remove(request)
                    del self.requests[request.id]
                    request.state = "rejected"
                    request.rejection = "client_canceled_before_submit" if cancel.is_set() else "worker_not_accepting"
                    request.items.append(("rejected", request.rejection))
                    if not self.pending:
                        self.pending_since = None
                    self.cond.notify_all()
                    break
                owners = 1 if self.owner_count is None else self.owner_count()
                P.require(type(owners) is int and 1 <= owners <= 2, "Existing bounded backend owner count required")
                if not self.unusable and self.group is None and self.pending and (
                        len(self.pending) == 2 or (len(self.pending) >= owners
                        and now - self.pending_since >= self.config.cohort_wait_seconds)):
                    self._submit_pending_locked()
                if request.state == "submitted" and deadline is None:
                    deadline = request.admission_deadline
                self._poll_cancel_locked()
                if deadline is not None and now >= deadline and request.state == "submitted":
                    self._fail_locked("Worker did not answer a cohort submission within the admission bound")
                    break
                if request.state in ("pending", "submitted"):
                    self.cond.wait(min(0.05, self.config.cohort_wait_seconds))
        return request  # Registered owners always return, even after a channel failure; _finish holds them to closure.

    def _submit_pending_locked(self):
        for request in tuple(self.pending):
            if request.client_cancel.is_set():
                self.pending.remove(request)
                del self.requests[request.id]
                request.state, request.rejection = "rejected", "client_canceled_before_submit"
                request.items.append(("rejected", request.rejection))
        if not self.pending:
            self.pending_since = None
            return
        pages = tuple(self.pending)
        P.require(1 <= len(pages) <= 2 and self.group is None, "Invalid local cohort capacity")
        ident = "g-" + secrets.token_hex(16)
        line = P.message("submit_cohort", id=ident, model="chandra", requests=[{
            "id": r.id, "input_manifest": r.prepared.manifest,
            "input_sha256": r.prepared.manifest_sha256, "output": r.output} for r in pages])
        admission_deadline = self.clock() + self.config.admission_seconds
        for slot, request in enumerate(pages):
            request.state = "submitted"
            request.admission_deadline = admission_deadline
            request.cohort_id, request.cohort_slot, request.cohort_page_count = ident, slot, len(pages)
        self.pending, self.pending_since = [], None
        self.group = {"id": ident, "pages": pages, "cancel_sent": False, "cancel_deadline": None, "aggregate_deadline": None, "started": False}
        self._send_locked(line)  # Failure keeps every owner registered until existing confirmed worker closure.

    def _poll_cancel_locked(self):
        group = self.group
        if group is None or self.failure or self.closed or self.exit_code is not None:
            return
        if all(r.cohort_terminal is not None for r in group["pages"]):
            if self.clock() > group["aggregate_deadline"]:
                self._fail_locked("Worker did not emit the aggregate retirement within the admission bound")
            return  # Native already emitted every page retirement; do not race an unnecessary cancel against the aggregate.
        if not group["cancel_sent"] and all(r.state != "submitted" for r in group["pages"]) and all(r.client_cancel.is_set() for r in group["pages"]):
            if self._send_locked(P.message("cancel", id=group["id"])):
                group["cancel_sent"] = True
                group["cancel_deadline"] = self.clock() + self.config.cancel_grace_seconds
                for request in group["pages"]:
                    request.cancel_sent = True
        if group["cancel_deadline"] is not None and self.clock() > group["cancel_deadline"]:
            group["cancel_deadline"] = None
            self._fail_locked("Whole-cohort cancellation was not drained within the grace bound")

    def _supervise_once(self):
        with self.cond:
            self._poll_cancel_locked()
        return super()._supervise_once()

    def next_item(self, request, cancel):
        with self.cond:
            while True:
                self._poll_cancel_locked()  # Individual client cancellation never emits a page cancel.
                if request.items:
                    return request.items.popleft()
                if self.closed:
                    return ("closed", None)
                self.cond.wait(0.05)

    def _mapped(self, event):
        ident = event.get("id")
        request = self.requests.get(ident) if P.text(ident) else None
        P.require(self.group is not None and request in self.group["pages"], "Event for an unknown cohort page")
        P.require(P.exact(event.get("cohort_id"), request.cohort_id)
                  and P.exact(event.get("cohort_slot"), request.cohort_slot), "Cohort page mapping changed")
        return request

    def _remember(self, ident, canceled):
        self.recent[ident] = canceled
        while len(self.recent) > RECENT:
            self.recent.popitem(last=False)

    def _dispatch(self, event):
        kind = event["event"]
        if self.ready is None or kind in ("closing", "exit"):
            return super()._dispatch(event)
        P.require(self.exit_event is None, "Event after exit")
        if kind == "admitted":
            request = self._mapped(event)
            P.require(request.state == "submitted" and event.get("slot") == "active_cohort"
                      and event.get("output") == request.output and event.get("input_sha256") == request.prepared.manifest_sha256
                      and event.get("diagnostic_token_cap") is None and P.exact(event.get("output_allowance"), P.OUTPUT_ALLOWANCE)
                      and P.integer(event.get("request_index"), 1), "Cohort admission commitments differ")
            request.state = "admitted"
            request.native_index = event["request_index"]
        elif kind == "cohort_started":
            group = self.group
            P.require(group is not None and P.exact(event.get("id"), group["id"]) and not group["started"]
                      and all(r.state == "admitted" for r in group["pages"])
                      and P.exact(event.get("slots"), len(group["pages"])) and event.get("mode") == "execute", "Invalid cohort start")
            group["started"] = True
            for request in group["pages"]:
                request.state = "started"
        elif kind in ("phase", "token"):
            self._mapped(event)
            super()._dispatch(event)  # Preserve original token order/vocabulary/EOS/allowance checks for each actual page.
        elif kind == "terminal":
            request = self._mapped(event)
            P.require(request.state in ("admitted", "started"), "Duplicate or premature page terminal")
            request.cohort_survivable, _ = P.check_terminal(event, request.tokens, self.resident)
            request.cohort_terminal = event
            request.state = "awaiting_cohort_retirement"
            if all(r.cohort_terminal is not None for r in self.group["pages"]):
                self.group["aggregate_deadline"] = self.clock() + self.config.admission_seconds
            # Do not set terminal or release the caller until the aggregate matches all page owners.
        elif kind == "cohort_terminal":
            group = self.group
            P.require(group is not None and P.exact(event.get("id"), group["id"]), "Unknown cohort terminal")
            pages = group["pages"]
            P.require(all(r.cohort_terminal is not None for r in pages), "Aggregate before every page terminal")
            expected = [{"id": r.id, "slot": r.cohort_slot, "status": r.cohort_terminal["status"],
                         "released": r.cohort_terminal["released"], "result_sha256": r.cohort_terminal.get("result_sha256")} for r in pages]
            actual = event.get("pages")
            P.require(isinstance(actual, list) and len(actual) == len(expected)
                      and all(isinstance(a, dict) and set(a) == set(b) and all(P.exact(a[k], v) for k, v in b.items())
                              for a, b in zip(actual, expected)), "Aggregate page owners/status/receipt commitments differ")
            released = all(r.cohort_terminal["released"] for r in pages)
            poisoned = any(r.cohort_terminal["worker_poisoned"] for r in pages)
            status = "failed" if any(r.cohort_terminal["status"] == "failed" for r in pages) else (
                "canceled" if any(r.cohort_terminal["status"] == "canceled" for r in pages) else "completed")
            P.require(P.exact(event.get("released"), released) and P.exact(event.get("worker_poisoned"), poisoned)
                      and P.exact(event.get("status"), status) and event.get("request_replayed") is False,
                      "Aggregate release/poison/replay commitments differ")
            if not all(r.cohort_survivable for r in pages):
                self.unusable = "Cohort did not prove every owner drained and released"
                self._retire_locked("unclean_cohort_release", force=False)
                return  # Owners stay registered until confirmed process/Job closure; no success or replay.
            for request in pages:
                request.cohort_released = True
                request.state, request.terminal = "terminal", request.cohort_terminal
                request.items.append(("terminal", request.terminal))
                self.counts[request.terminal["status"]] += 1
                del self.requests[request.id]
                self._remember(request.id, False)  # No page cancellation is ever sent.
            self._remember(group["id"], group["cancel_sent"])
            self.group = None
            self.pending_since = self.clock() if self.pending else None
        elif kind == "rejected":
            group = self.group
            P.require(group is not None and P.exact(event.get("submitted_id"), group["id"])
                      and all(r.state == "submitted" for r in group["pages"]) and P.text(event.get("reason")), "Invalid cohort rejection")
            for request in group["pages"]:
                request.state, request.rejection = "rejected", event["reason"]
                request.items.append(("rejected", request.rejection))
                del self.requests[request.id]
            self.group = None
        elif kind == "cancel_requested":
            group = self.group
            P.require(group is not None and P.exact(event.get("id"), group["id"]) and event.get("released") is False
                      and event.get("state") in ("admitted", "active") and P.text(event.get("origin")), "Invalid group cancellation acknowledgement")
            if event["origin"] == "client":
                P.require(group["cancel_sent"], "Unsent group cancellation")
        elif kind == "cancel_rejected":
            submitted = event.get("submitted_id")
            P.require(event.get("reason") == "already_terminal" and P.text(submitted) and self.recent.get(submitted) is True,
                      "Group cancellation refused unexpectedly")
        else:
            raise P.ProtocolViolation("Unexpected event in dedicated cohort mode: " + kind)
