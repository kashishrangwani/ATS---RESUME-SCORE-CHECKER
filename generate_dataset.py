import os
import random
import pandas as pd

# ------- Create folders -------
os.makedirs("data/resumes", exist_ok=True)

# ------- Sample pool for auto generation -------
skills_pool = [
    "Python", "Java", "JavaScript", "SQL", "HTML", "CSS", "Machine Learning",
    "Deep Learning", "Data Analysis", "Communication", "Teamwork",
    "Problem Solving", "Leadership", "C++", "Cloud", "AWS"
]

experience_pool = [
    "Worked on web development projects",
    "Internship in data analytics",
    "Created machine learning models",
    "Handled database management",
    "Worked on frontend UI",
    "API integration work",
    "Cloud deployment using AWS",
    "Built automation scripts"
]

education_pool = [
    "B.Tech in Computer Science",
    "BCA",
    "MCA",
    "BSc IT",
    "Diploma in Software Engineering"
]

labels = ["Good", "Average", "Poor"]

# ------- Generate 50 resumes -------
resume_data = []

for i in range(1, 51):
    file_name = f"resume_{i}.txt"
    path = f"data/resumes/{file_name}"

    selected_skills = ", ".join(random.sample(skills_pool, random.randint(5, 10)))
    selected_exp = ", ".join(random.sample(experience_pool, random.randint(2, 4)))
    selected_edu = random.choice(education_pool)

    resume_text = f"""
    Name: Candidate {i}
    Education: {selected_edu}
    Skills: {selected_skills}
    Experience: {selected_exp}
    """

    with open(path, "w", encoding="utf-8") as f:
        f.write(resume_text)

    # Assign random labels
    assigned_label = random.choice(labels)
    resume_data.append([file_name, assigned_label])


# ------- Save labels.csv -------
df = pd.DataFrame(resume_data, columns=["file_name", "label"])
df.to_csv("data/labels.csv", index=False)

print("✔ 50 synthetic resumes generated successfully!")
print("✔ labels.csv created!")
print("✔ Dataset ready for ML training!")
