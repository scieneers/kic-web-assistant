import os
import sys
import json
import logging
from datetime import timedelta

import azure.functions as func
import azure.durable_functions as df

# Sentinel embedded in the activity's failure message when Moodle is in
# maintenance mode. The orchestrator string-matches on this to decide whether to
# wait-and-retry (vs. failing immediately on any other error). It is independent
# of the exception class so it survives Durable's serialization and the differing
# import roots between the activity and the loaders package.
MOODLE_MAINTENANCE_SENTINEL = "MOODLE_MAINTENANCE"


def _is_maintenance_error(exc: BaseException) -> bool:
    """True if exc (or anything in its cause/context chain) is a Moodle site-down error.

    Matched by class name rather than import to avoid coupling to the loaders
    package's deploy path.
    """
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ == "MoodleMaintenanceError":
            return True
        cur = cur.__cause__ or cur.__context__
    return False

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from get_data import Fetch_Data
from run_logger import RunLogger, create_run_log_sas_url

app = func.FunctionApp()


# =============================================================================
# DURABLE FUNCTIONS PATTERN
# =============================================================================

# 1. HTTP Starter - Starts the orchestration and returns status URLs
@app.function_name(name="start_ingest")
@app.route(route="start_ingest", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
@app.durable_client_input(client_name="client")
async def start_ingest(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    """Start the durable data ingestion orchestration."""
    # Reject overlapping runs: start with a deterministic singleton instance_id.
    # Durable will throw if an instance with that id already exists.
    singleton_instance_id = "ingest_singleton"

    # If a previous instance is still running, reject.
    existing = await client.get_status(singleton_instance_id)
    if existing and existing.runtime_status and existing.runtime_status.name in {"Running", "Pending"}:
        return func.HttpResponse(
            json.dumps(
                {
                    "error": "INGEST_ALREADY_RUNNING",
                    "instanceId": singleton_instance_id,
                    "runtimeStatus": existing.runtime_status.name,
                    "statusQueryGetUri": client.create_check_status_response(req, singleton_instance_id).headers.get(
                        "Location"
                    ),
                }
            ),
            status_code=409,
            mimetype="application/json",
        )

    # If a previous singleton instance completed/failed/terminated, purge its history so we can
    # re-use the deterministic instance id.
    if existing and existing.runtime_status and existing.runtime_status.name in {"Completed", "Failed", "Terminated"}:
        try:
            await client.purge_instance_history(singleton_instance_id)
        except Exception:
            # best-effort; Durable may already have purged or not support purge in this context
            pass

    # Generate a run_id now so we can expose a log URL immediately.
    run_id = RunLogger.new_run_id()
    log_url = create_run_log_sas_url(run_id, expiry_hours=24)

    instance_id = await client.start_new(
        "ingest_orchestrator",
        singleton_instance_id,
        {"run_id": run_id, "log_url": log_url},
    )
    logging.info(f"Started orchestration with ID = '{instance_id}' (run_id={run_id})")

    # Return the standard Durable status URLs + our run metadata so the caller can
    # open the log URL while the run is still executing.
    status_response = client.create_check_status_response(req, instance_id)
    
    try:
        urls = json.loads(status_response.get_body().decode('utf-8'))
    except Exception:
        urls = {}

    body = {
        "instanceId": instance_id,
        "run_id": run_id,
        "log_url": log_url,
        "statusQueryGetUri": urls.get("statusQueryGetUri") or status_response.headers.get("Location"),
        "sendEventPostUri": urls.get("sendEventPostUri"),
        "terminatePostUri": urls.get("terminatePostUri"),
        "purgeHistoryDeleteUri": urls.get("purgeHistoryDeleteUri"),
    }
    return func.HttpResponse(
        json.dumps(body), 
        status_code=202, 
        headers=status_response.headers,
        mimetype="application/json"
    )


# 2. Orchestrator - Coordinates the workflow, survives restarts
@app.function_name(name="ingest_orchestrator")
@app.orchestration_trigger(context_name="context")
def ingest_orchestrator(context: df.DurableOrchestrationContext):
    """Orchestrator that coordinates data ingestion activities."""
    payload = context.get_input() or {}

    # Retry ONLY when Moodle is in maintenance mode (the whole site is down).
    # Any other failure surfaces immediately. We can't use call_activity_with_retry
    # for this because its RetryOptions retries on *any* exception; instead we loop
    # with a durable timer and re-dispatch only on the maintenance sentinel.
    # Reading env here is replay-safe (deterministic within a run).
    retry_interval_min = int(os.getenv("INGEST_RETRY_INTERVAL_MINUTES", "30"))
    retry_max_attempts = int(os.getenv("INGEST_RETRY_MAX_ATTEMPTS", "6"))

    attempt = 0
    while True:
        attempt += 1
        try:
            result = yield context.call_activity("run_data_extraction", payload)
            return result
        except Exception as e:
            is_maintenance = MOODLE_MAINTENANCE_SENTINEL in str(e)
            if is_maintenance and attempt < retry_max_attempts:
                # Site is down — wait via a durable timer, then re-dispatch.
                deadline = context.current_utc_datetime + timedelta(minutes=retry_interval_min)
                yield context.create_timer(deadline)
                continue
            # Non-maintenance failure, or maintenance retries exhausted: fail the run.
            raise


# 3. Activity Function - Does the actual work
@app.function_name(name="run_data_extraction")
@app.activity_trigger(input_name="payload")
def run_data_extraction(payload) -> str:
    """Activity that performs the actual data extraction."""
    logger = logging.getLogger("loader")
    logger.info("Starting data extraction activity...")
    try:
        run_id = None
        preset_log_url = None
        if isinstance(payload, dict):
            run_id = payload.get("run_id")
            preset_log_url = payload.get("log_url")

        logger.info("Activity payload received (run_id=%s)", run_id)
        result = Fetch_Data(run_id=run_id, preset_log_url=preset_log_url).extract()
        # Durable Functions will JSON-serialize dict outputs.
        return result
    except Exception as e:
        if _is_maintenance_error(e):
            # Tag the failure so the orchestrator waits-and-retries instead of
            # failing the run. The whole Moodle site is down; retrying later is
            # the right move, and continuing would risk deleting indexed data.
            logger.warning("Moodle site in maintenance mode; signaling orchestrator to retry: %s", e)
            raise Exception(f"{MOODLE_MAINTENANCE_SENTINEL}: {e}") from e
        logger.exception("Data extraction failed")
        raise


# 4. Status Check Endpoint - Check orchestration status
@app.function_name(name="check_status")
@app.route(route="check_status/{instance_id}", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
@app.durable_client_input(client_name="client")
async def check_status(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    """Check the status of a running orchestration."""
    instance_id = req.route_params.get("instance_id")
    status = await client.get_status(instance_id)
    if status:
        return func.HttpResponse(
            json.dumps({
                "instanceId": status.instance_id,
                "runtimeStatus": status.runtime_status.name,
                "output": status.output,
                "createdTime": str(status.created_time),
                "lastUpdatedTime": str(status.last_updated_time),
            }),
            mimetype="application/json"
        )
    return func.HttpResponse("Instance not found", status_code=404)


# 5. Stop Endpoint - Manually terminate the orchestration
@app.function_name(name="stop_ingest")
@app.route(route="stop_ingest/{instance_id}", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
@app.durable_client_input(client_name="client")
async def stop_ingest(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    """Manually terminate a running durable data ingestion orchestration."""
    instance_id = req.route_params.get("instance_id")
    if not instance_id:
        # Default to the singleton instance ID if not provided in route
        instance_id = "ingest_singleton"

    status = await client.get_status(instance_id)
    if not status or status.runtime_status.name not in {"Running", "Pending"}:
        return func.HttpResponse(f"Instance '{instance_id}' is not currently running.", status_code=400)

    reason = req.params.get("reason") or "Terminated by user request"
    await client.terminate(instance_id, reason)
    
    return func.HttpResponse(
        f"Termination request sent for instance '{instance_id}'.",
        status_code=202
    )


# =============================================================================
# LEGACY TRIGGERS (kept for backwards compatibility)
# =============================================================================

@app.function_name(name="timer_trigger")
# @app.timer_trigger(schedule="0 0 5 * * 4", arg_name="mytimer", run_on_startup=False, use_monitor=False)
@app.timer_trigger(schedule="0 0 5 * * *", arg_name="mytimer", run_on_startup=False, use_monitor=False)
@app.durable_client_input(client_name="client")
async def timer_trigger(mytimer: func.TimerRequest, client: df.DurableOrchestrationClient) -> None:
    # """Weekly timer that starts the durable orchestration."""
    """Daily timer that starts the durable orchestration."""
    singleton_instance_id = "ingest_singleton"
    existing = await client.get_status(singleton_instance_id)
    if existing and existing.runtime_status and existing.runtime_status.name in {"Running", "Pending"}:
        logging.warning(
            "Timer ingest skipped because a run is already active (instanceId=%s status=%s)",
            singleton_instance_id,
            existing.runtime_status.name,
        )
        return

    if existing and existing.runtime_status and existing.runtime_status.name in {"Completed", "Failed", "Terminated"}:
        try:
            await client.purge_instance_history(singleton_instance_id)
        except Exception:
            pass

    run_id = RunLogger.new_run_id()
    log_url = create_run_log_sas_url(run_id, expiry_hours=24)
    instance_id = await client.start_new("ingest_orchestrator", singleton_instance_id, {"run_id": run_id, "log_url": log_url})
    logging.info(f"Timer started orchestration with ID = '{instance_id}' (run_id={run_id})")


@app.function_name(name="manual_trigger")
@app.route(route="manual_ingest", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def manual_trigger(req: func.HttpRequest) -> func.HttpResponse:
    """Legacy manual trigger - consider using /api/start_ingest instead."""
    try:
        Fetch_Data().extract()
        return func.HttpResponse("Data ingestion completed successfully!", status_code=200)
    except Exception as e:
        return func.HttpResponse(f"Data ingestion failed: {str(e)}", status_code=500)
