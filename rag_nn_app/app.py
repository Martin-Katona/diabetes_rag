from tkinter import *
import customtkinter

from PIL import Image, ImageTk

import sys
from io import StringIO


from diag_models.diabetes.diabetes_predictor import diabetes_predict,explain_prediction,scale_input

from rag.rag_01 import create_response_and_sources, print_formatted_output, process_query, query_llm,create_disease_prediction_string,create_queries_list

# Main Window Properties
window = Tk()
window.title("Diabetes Prediction")
window.geometry("500x700")
window.configure(bg="#D3D3D3")  # Set window background to a slightly darker grey

# Gender Radio Buttons
gender_var = StringVar(value="None")

Label(window, text="Sex:", font=("Arial", 14), bg="#D3D3D3", fg="#000000").place(x=60, y=50)
male_button = customtkinter.CTkRadioButton(
    master=window, text="Male", variable=gender_var, value="Male", bg_color="#D3D3D3", text_color="#000000"
)
male_button.place(x=120, y=50)

female_button = customtkinter.CTkRadioButton(
    master=window, text="Female", variable=gender_var, value="Female", bg_color="#D3D3D3", text_color="#000000"
)
female_button.place(x=200, y=50)

# Input Fields
age_entry = Entry(window, width=10, font=("Arial", 14), bg="#FFFFFF", fg="#000000", highlightthickness=0)
age_entry.place(x=250, y=100)
Label(window, text="Age:", font=("Arial", 14), bg="#D3D3D3", fg="#000000").place(x=60, y=100)

weight_entry = Entry(window, width=10, font=("Arial", 14), bg="#FFFFFF", fg="#000000", highlightthickness=0)
weight_entry.place(x=250, y=150)
Label(window, text="Weight (kg):", font=("Arial", 14), bg="#D3D3D3", fg="#000000").place(x=60, y=150)

height_entry = Entry(window, width=10, font=("Arial", 14), bg="#FFFFFF", fg="#000000", highlightthickness=0)
height_entry.place(x=250, y=200)
Label(window, text="Height (cm):", font=("Arial", 14), bg="#D3D3D3", fg="#000000").place(x=60, y=200)

mental_health_entry = Entry(window, width=10, font=("Arial", 14), bg="#FFFFFF", fg="#000000", highlightthickness=0)
mental_health_entry.place(x=250, y=265)
Label(
    window,
    text="Days of poor mental health \nin the last 30 days:",
    font=("Arial", 14),
    bg="#D3D3D3",
    fg="#000000",
    justify="left",
).place(x=60, y=250)

# Checkboxes
Label_id7 = customtkinter.CTkLabel(
    master=window,
    text="Check the box if you agree with the statement",
    font=("Arial", 14),
    text_color="#000000",
    height=30,
    width=391,
    corner_radius=0,
    bg_color="#D3D3D3",
    fg_color="#D3D3D3",
)
Label_id7.place(x=9, y=300)

Checkbox_id11 = customtkinter.CTkCheckBox(
    master=window,
    text="I have high cholesterol.",
    text_color="#000000",
    border_color="#000000",
    fg_color="#808080",
    hover_color="#808080",
    corner_radius=4,
    border_width=2,
)
Checkbox_id11.place(x=60, y=340)

Checkbox_id16 = customtkinter.CTkCheckBox(
    master=window,
    text="I have high blood pressure.",
    text_color="#000000",
    border_color="#000000",
    fg_color="#808080",
    hover_color="#808080",
    corner_radius=4,
    border_width=2,
)
Checkbox_id16.place(x=60, y=370)

Checkbox_id13 = customtkinter.CTkCheckBox(
    master=window,
    text="I have engaged in physical activity in the past 30 days.",
    text_color="#000000",
    border_color="#000000",
    fg_color="#808080",
    hover_color="#808080",
    corner_radius=4,
    border_width=2,
)
Checkbox_id13.place(x=60, y=400)

Checkbox_id12 = customtkinter.CTkCheckBox(
    master=window,
    text="I have smoked at least 100 cigarettes in my entire life.",
    text_color="#000000",
    border_color="#000000",
    fg_color="#808080",
    hover_color="#808080",
    corner_radius=4,
    border_width=2,
)
Checkbox_id12.place(x=60, y=430)

Checkbox_id14 = customtkinter.CTkCheckBox(
    master=window,
    text="I consume fruits or vegetables one or more times per day.",
    text_color="#000000",
    border_color="#000000",
    fg_color="#808080",
    hover_color="#808080",
    corner_radius=4,
    border_width=2,
)
Checkbox_id14.place(x=60, y=460)

Checkbox_id15 = customtkinter.CTkCheckBox(
    master=window,
    text="I consume heavy alcohol (for men: 14 or more drinks per week;\nfor women: 7 or more drinks per week).",
    text_color="#000000",
    border_color="#000000",
    fg_color="#808080",
    hover_color="#808080",
    corner_radius=4,
    border_width=2,
)
Checkbox_id15.place(x=60, y=490)


# Function to open a new window
def open_new_window():

        # Calculate BMI
    weight = float(weight_entry.get())
    height = float(height_entry.get()) / 100  # Convert height from cm to m
    bmi = weight / (height ** 2)

    # Calculate age group
    age = float(age_entry.get())
    if age <= 24:
        age_group = 1
    elif age <= 29:
        age_group = 2
    elif age <= 34:
        age_group = 3
    elif age <= 39:
        age_group = 4
    elif age <= 44:
        age_group = 5
    elif age <= 49:
        age_group = 6
    elif age <= 54:
        age_group = 7
    elif age <= 59:
        age_group = 8
    elif age <= 64:
        age_group = 9
    elif age <= 69:
        age_group = 10
    elif age <= 74:
        age_group = 11
    elif age <= 79:
        age_group = 12
    else:
        age_group = 13


    input_features = {
    "Age": age_group,
    "Sex": 1 if gender_var.get() == "Male" else 0,
    "HighChol": 1 if Checkbox_id11.get() else 0,  # CHECKBOX VALUES MUST BE INT
    "BMI": bmi,
    "Smoker": 1 if Checkbox_id12.get() else 0,  # CHECKBOX VALUES MUST BE INT
    "PhysActivity": 1 if Checkbox_id13.get() else 0,  # CHECKBOX VALUES MUST BE INT
    "Fruits": 1 if Checkbox_id14.get() else 0,  # CHECKBOX VALUES MUST BE INT
    "HvyAlcoholConsump": 1 if Checkbox_id15.get() else 0,  # CHECKBOX VALUES MUST BE INT
    "MentHlth": float(mental_health_entry.get()),
    "HighBP": 1 if Checkbox_id16.get() else 0   # CHECKBOX VALUES MUST BE INT
    }



    new_window = Toplevel(window)
    new_window.title("Survey Results")
    new_window.geometry("1700x1000")
    new_window.configure(bg="#D3D3D3")


    # Left frame for text output
    left_frame = Frame(new_window, bg="#D3D3D3")
    left_frame.place(x=0, y=0, width=800, height=1000)

    # Right frame for additional questions and response
    right_frame = Frame(new_window, bg="#D3D3D3")
    right_frame.place(x=800, y=0, width=890, height=550)  # Top right half

    Label(left_frame, text="Result Analysis", font=("Arial", 14), bg="#D3D3D3", fg="#000000").pack(pady=20)
    text_output = Text(left_frame, font=("Arial", 12), bg="#FFFFFF", fg="#000000",wrap="word")
    text_output.pack(fill=BOTH, expand=True, padx=10, pady=10)

    # Additional questions and response area (right frame)
    Label(right_frame, text="Additional Questions:", font=("Arial", 14), bg="#D3D3D3", fg="#000000").pack(pady=20)

    # Frame to hold entry and submit button
    question_frame = Frame(right_frame, bg="#D3D3D3")
    question_frame.pack(pady=10)

    question_entry = Entry(question_frame, width=50, font=("Arial", 14), bg="#FFFFFF", fg="#000000")
    question_entry.pack(side=LEFT, padx=5)


    # Create a new frame for the bottom right half
    bottom_right_frame = Frame(new_window, bg="#D3D3D3")
    bottom_right_frame.place(x=800, y=550, width=900, height=450)  # Adjusted for a 1700x1000 window


    # Add a caption for the first image

    Label(right_frame, text="Response:", font=("Arial", 14), bg="#D3D3D3", fg="#000000").pack(pady=10)
    response_text = Text(right_frame, font=("Arial", 12), bg="#FFFFFF", fg="#000000",wrap="word")
    response_text.pack(fill=BOTH, expand=True, padx=10, pady=10)


    # Generate and display predictions
    try:
        # Process predictions
        prediction_probability = diabetes_predict(scale_input(input_features))
        shap_values, shap_values_for_plot = explain_prediction(scale_input(input_features))
        diagnosis_query = create_disease_prediction_string(
            shap_values.values[0], 
            shap_values_for_plot.feature_names, 
            prediction_probability, 
            'diabetes', 
            input_features
        )
        queries_list = create_queries_list(
            shap_values.values[0], 
            shap_values_for_plot.feature_names, 
            'diabetes'
        )

        # Get LLM response
        model = "mistral"
        response = query_llm(diagnosis_query, model, template_type='diagnosis')
         # Insert main response first

        text_output.insert(END, f"Prediction : {int(prediction_probability*100)}% probability of having Diabetes\n\n")
        text_output.insert(END, f"Analysis:\n{response}\n\n")
        
        # Get formatted explanations
        model = "mistral"
        emb_model = "nomic-embed-text"
        responses_dict, sources_dict = create_response_and_sources(
            queries_list, process_query, model, emb_model
        )
        
        # Redirect print output to text widget
        buffer = StringIO()
        original_stdout = sys.stdout
        sys.stdout = buffer
        
        print_formatted_output(responses_dict, sources_dict)
        
        sys.stdout = original_stdout
        
        # Insert into text widget
        formatted_text = buffer.getvalue()
        text_output.insert(END, formatted_text)

            # Load your images

        image1_path = "diag_models/diabetes/diabetes/plots/dia_bar_plot.png"
        image2_path = "diag_models/diabetes/diabetes/plots/dia_waterfall_plot.png"

        image1 = Image.open(image1_path)
        image1.thumbnail((420, 420))  # Resize to fit half the frame
        image1_tk = ImageTk.PhotoImage(image1)

        image1_caption = Label(bottom_right_frame, text="Feature importance for the prediction \n(blue: anti-diabetes, red: pro-diabetes)", font=("Arial", 12), bg="#D3D3D3", fg="#000000")
        image1_caption.place(x=10, y=350)

        image2 = Image.open(image2_path)
        image2.thumbnail((420, 420))  # Resize to fit half the frame
        image2_tk = ImageTk.PhotoImage(image2)

        image2_caption = Label(bottom_right_frame, text="The cumulative effect of all features contributing to prediction \n(blue: anti-diabetes, red: pro-diabetes)", font=("Arial", 12), bg="#D3D3D3", fg="#000000")
        image2_caption.place(x=460, y=350)

        # Create labels for the images
        image1_label = Label(bottom_right_frame, image=image1_tk)
        image1_label.image = image1_tk  # Keep a reference to prevent garbage collection
        image1_label.place(x=10, y=0)

        image2_label = Label(bottom_right_frame, image=image2_tk)
        image2_label.image = image2_tk  # Keep a reference to prevent garbage collection
        image2_label.place(x=460, y=0)
        
    except Exception as e:
        text_output.insert(END, f"Error generating report: {str(e)}")

    # Function to handle submit button click
    def submit_question():
        query = question_entry.get()
        if query:
            # Process query
            model = "mistral"
            emb_model = "nomic-embed-text"
            
            # Create a list with the query
            queries_list = [query]
            
            # Get responses and sources
            responses_dict, sources_dict = create_response_and_sources(
                queries_list, process_query, model, emb_model,fusion="true"
            )
            
            # Redirect print output to text widget
            buffer = StringIO()
            original_stdout = sys.stdout
            sys.stdout = buffer
            
            print_formatted_output(responses_dict, sources_dict)
            
            sys.stdout = original_stdout
            
            # Insert into response text widget
            formatted_text = buffer.getvalue()
            response_text.delete('1.0', END)  # Clear previous responses
            response_text.insert(END, formatted_text)
        else:
            response_text.insert(END, "Please enter a question.\n")

    submit_button = customtkinter.CTkButton(
        master=question_frame,
        text="Submit",
        font=("undefined", 14),
        text_color="#000000",
        hover=True,
        hover_color="#949494",
        height=30,
        width=100,
        border_width=2,
        corner_radius=6,
        border_color="#000000",
        bg_color="#D3D3D3",
        fg_color="#F0F0F0",
        command=submit_question,
    )
    submit_button.pack(side=LEFT, padx=5)


# TEST FUNCTION
    # Function to open a new window
def open_new_window_test():
    new_window = Toplevel(window)
    new_window.title("Survey Results")
    new_window.geometry("1700x1000")
    new_window.configure(bg="#D3D3D3")




    # Left frame for text output
    left_frame = Frame(new_window, bg="#D3D3D3")
    left_frame.place(x=0, y=0, width=800, height=1000)

    # Right frame for additional questions and response
    right_frame = Frame(new_window, bg="#D3D3D3")
    right_frame.place(x=800, y=0, width=900, height=550)  # Top right half

    Label(left_frame, text="Text Output Area", font=("Arial", 14), bg="#D3D3D3", fg="#000000").pack(pady=20)
    text_output = Text(left_frame, font=("Arial", 12), bg="#FFFFFF", fg="#000000")
    text_output.pack(fill=BOTH, expand=True, padx=10, pady=10)

    # Additional questions and response area (right frame)
    Label(right_frame, text="Additional Questions:", font=("Arial", 14), bg="#D3D3D3", fg="#000000").pack(pady=20)

    # Frame to hold entry and submit button
    question_frame = Frame(right_frame, bg="#D3D3D3")
    question_frame.pack(pady=10)

    question_entry = Entry(question_frame, width=50, font=("Arial", 14), bg="#FFFFFF", fg="#000000")
    question_entry.pack(side=LEFT, padx=5)


    Label(right_frame, text="Response:", font=("Arial", 14), bg="#D3D3D3", fg="#000000").pack(pady=10)
    response_text = Text(right_frame, font=("Arial", 12), bg="#FFFFFF", fg="#000000")
    response_text.pack(fill=BOTH, expand=True, padx=10, pady=10)

    # Create a new frame for the bottom right half
    bottom_right_frame = Frame(new_window, bg="#D3D3D3")
    bottom_right_frame.place(x=800, y=550, width=900, height=450)  # Adjusted for a 1700x1000 window

    # Load your images

    image1_path = "diag_models/diabetes/diabetes/plots/dia_bar_plot.png"
    image2_path = "diag_models/diabetes/diabetes/plots/dia_waterfall_plot.png"

    image1 = Image.open(image1_path)
    image1.thumbnail((420, 420))  # Resize to fit half the frame
    image1_tk = ImageTk.PhotoImage(image1)

    image1_caption = Label(bottom_right_frame, text="Feature importance for the prediction \n(blue: anti-diabetes, red: pro-diabetes)", font=("Arial", 12), bg="#D3D3D3", fg="#000000")
    image1_caption.place(x=10, y=350)

    image2 = Image.open(image2_path)
    image2.thumbnail((420, 420))  # Resize to fit half the frame
    image2_tk = ImageTk.PhotoImage(image2)

    image2_caption = Label(bottom_right_frame, text="The cumulative effect of all features contributing to prediction \n(blue: anti-diabetes, red: pro-diabetes)", font=("Arial", 12), bg="#D3D3D3", fg="#000000")
    image2_caption.place(x=460, y=350)

    # Create labels for the images
    image1_label = Label(bottom_right_frame, image=image1_tk)
    image1_label.image = image1_tk  # Keep a reference to prevent garbage collection
    image1_label.place(x=10, y=0)

    image2_label = Label(bottom_right_frame, image=image2_tk)
    image2_label.image = image2_tk  # Keep a reference to prevent garbage collection
    image2_label.place(x=460, y=0)


    # Function to handle submit button click
    def submit_question():
        question = question_entry.get()
        response_text.insert(END, f"Question: {question}\nResponse: This is a placeholder response.\n\n")

    submit_button = customtkinter.CTkButton(
        master=question_frame,
        text="Submit",
        font=("undefined", 14),
        text_color="#000000",
        hover=True,
        hover_color="#949494",
        height=30,
        width=100,
        border_width=2,
        corner_radius=6,
        border_color="#000000",
        bg_color="#FFFFFF",
        fg_color="#F0F0F0",
        command=submit_question,
    )
    submit_button.pack(side=LEFT, padx=5)

# Submit button
submit_button = customtkinter.CTkButton(
    master=window,
    text="Submit",
    font=("undefined", 14),
    text_color="#000000",
    hover=True,
    hover_color="#949494",
    height=30,
    width=150,
    border_width=2,
    corner_radius=6,
    border_color="#000000",
    bg_color="#D3D3D3",
    fg_color="#F0F0F0",
    command=open_new_window,
)
submit_button.place(x=60, y=550)

# Run the main loop
window.mainloop()
