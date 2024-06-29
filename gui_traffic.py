import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

count_model = load_model("C:/Users/DELL/null_class/traffic_model/vehicle/Vehicle_Count_Model.keras")
color_model = load_model("C:/Users/DELL/null_class/traffic_model/vehicle/car_color_detection_model.keras")
pedestrian_model = load_model("C:/Users/DELL/null_class/traffic_model/vehicle/pedestrian_detection.keras")
gender_model = load_model("C:/Users/DELL/null_class/gender_age/final_model/Age_Sex_Detection_Model.keras")

def detect_cars_count_and_colors(image):
    resized_image = cv2.resize(image, (224, 224))
    resized_image = resized_image / 255.0  
    car_count = int(np.round(count_model.predict(np.expand_dims(resized_image, axis=0))[0][0]))
    other_vehicles_count = int(np.round(count_model.predict(np.expand_dims(resized_image, axis=0))[0][1]))
    if car_count == 0:
        predicted_color = 'None'
    else:
        color_prediction = color_model.predict(np.expand_dims(resized_image, axis=0))[0]
        predicted_color = 'Red' if color_prediction[0] > 0.5 else 'Blue'
        
    return car_count, other_vehicles_count, predicted_color

def detect_pedestrians_and_gender(image):
    resized_image_pedestrian = cv2.resize(image, (224, 224))
    resized_image_pedestrian = resized_image_pedestrian / 255.0  
    resized_image_pedestrian = np.expand_dims(resized_image_pedestrian, axis=0)
    pedestrian_prediction = pedestrian_model.predict(resized_image_pedestrian)[0]

    pedestrian_count = 0
    confidence_threshold = 0.5
    for i in range(5):
        class_prob = pedestrian_prediction[i, 4]
        if class_prob > confidence_threshold:
            pedestrian_count += 1

    gender_predictions = []
    for i in range(pedestrian_count):
        resized_image_gender = cv2.resize(image, (48, 48))
        resized_image_gender = resized_image_gender / 255.0  
        resized_image_gender = np.expand_dims(resized_image_gender, axis=0)
        gender_prediction = gender_model.predict(resized_image_gender)
        gender = 'Male' if gender_prediction[0][0] > 0.5 else 'Female'
        gender_predictions.append(gender)

    return pedestrian_count, gender_predictions

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image_pil = Image.fromarray(image)
        image_pil.thumbnail((400, 400))  
        img_tk = ImageTk.PhotoImage(image_pil)
        img_label.config(image=img_tk)
        img_label.image = img_tk
        global uploaded_image
        uploaded_image = cv2.imread(file_path)
        detect_button.config(state=tk.NORMAL) 
    else:
        messagebox.showerror("Error", "No file selected.")

def detect_features():
    global uploaded_image
    if uploaded_image is not None:
        car_count, other_vehicles_count, predicted_color = detect_cars_count_and_colors(uploaded_image)
        pedestrian_count, predicted_gender = detect_pedestrians_and_gender(uploaded_image)
        result_text = (
            f"Car Count: {car_count}\n"
            f"Other Vehicles Count: {other_vehicles_count}\n"
            f"Car Color: {predicted_color}\n"
            f"Pedestrian Count: {pedestrian_count}\n"
            f"Pedestrian Gender: {predicted_gender}"
        )
        messagebox.showinfo("Detection Results", result_text)
    else:
        messagebox.showerror("Error", "No image uploaded. Please upload an image first.")

root = tk.Tk()
root.title("Traffic Analysis Model")
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)
detect_button = tk.Button(root, text="Detect Features", command=detect_features, state=tk.DISABLED)
detect_button.pack(pady=10)
img_label = tk.Label(root)
img_label.pack()

uploaded_image = None
root.mainloop()