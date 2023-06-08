from datetime import datetime

import firebase_admin
from firebase_admin import credentials, db

from heatmap.heatmap import reset_heatmatrix

cred = credentials.Certificate(r"C:\Users\LENOVO\Desktop\Test\Yolov7-Sort-Tracking-Person\firebase\config.json")
firebase_admin.initialize_app(
    cred, {"databaseURL": "https://realtime-crud-e8421-default-rtdb.asia-southeast1.firebasedatabase.app/"})


def set_time_and_num_track(num_track):

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
    current_time = current_time.split(" ")
    date = current_time[0].split("-")[0] + current_time[0].split("-")[1] + current_time[0].split("-")[2]
    hour = current_time[1].split(":")[0] + current_time[1].split(":")[1]
    hour = hour + current_time[1].split(":")[2].split(".")[0]
    time = date + hour
    ref = db.reference("/time/" + time)
    ref.set(num_track)


def get_heatmap(heatmap):
    ref = db.reference("/heatmap").get()
    if ref == "True" and heatmap == False:
        reset_heatmatrix()
        return True
    elif ref == "False":
        return False
    else:
        return True
