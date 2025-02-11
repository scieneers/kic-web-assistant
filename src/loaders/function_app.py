import os
import sys

import azure.functions as func

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from get_data import Fetch_Data

app = func.FunctionApp()


@app.timer_trigger(schedule="0 0 5 * * 4", arg_name="mytimer", run_on_startup=True, use_monitor=False)
def timer_trigger(mytimer: func.TimerRequest) -> None:
    Fetch_Data().extract()
