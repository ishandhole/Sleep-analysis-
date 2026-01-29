# Sleep-Trend-Analyser: 

# 🛌 Sleep Trend Analyser

A comprehensive **Python-based Sleep Trend Analyser** that tracks, analyzes, and visualizes sleep patterns over time.
The system evaluates sleep duration, efficiency, latency, awakenings, and sleep stages to identify trends, predict sleep quality, and generate actionable insights.

---

## 📌 Features

* 📊 **Sleep Data Generation**
  Simulates realistic sleep data based on user input and natural variations.

* 🧹 **Data Preprocessing**
  Cleans data, handles missing values, and creates useful features like sleep debt and quality scores.

* 📈 **Trend & Pattern Analysis**

  * Identifies improving, worsening, or stable sleep trends
  * Compares weekday vs weekend sleep patterns

* 🧠 **Sleep Quality Prediction**
  Predicts nightly sleep quality scores (0–100) based on multiple factors.

* 📉 **Visual Analytics Dashboard**
  Interactive charts for:

  * Sleep duration trends
  * Sleep efficiency distribution
  * Latency vs duration
  * Average sleep by day of the week

* 📝 **Automated Sleep Report**
  Generates a detailed summary with metrics, insights, and recommendations.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Datetime**

---

## 📂 Project Structure

```
sleep_trend_analyser/
│
├── sleep_analysis_complete.py   # Main application file
├── README.md                    # Project documentation
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/sleep-trend-analyser.git
cd sleep-trend-analyser
```

### 2️⃣ Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn
```

### 3️⃣ Run the Program

```bash
python sleep_analysis_complete.py
```

---

## 🧪 How It Works

1. User enters basic sleep information (bedtime, wake time, interruptions, latency)
2. The system generates 30 days of sleep data
3. Data is preprocessed and enhanced with new features
4. Sleep metrics and trends are calculated
5. Sleep quality is predicted
6. Visual dashboards and a detailed report are generated

---

## 📊 Key Metrics Analyzed

* Sleep duration (hours)
* Sleep efficiency (%)
* Sleep latency (minutes)
* Night awakenings
* REM, deep, and light sleep duration
* Sleep debt and cumulative sleep debt
* Weekday vs weekend sleep differences

---

## 📌 Sample Insights Generated

* Detection of improving or worsening sleep patterns
* Identification of inconsistent sleep schedules
* Alerts for declining sleep quality
* Recommendations for better sleep hygiene

---

## 🎯 Use Cases

* Academic projects
* Data analytics practice
* Health-tech prototypes
* Personal sleep habit analysis
* Portfolio project for internships and placements

---

## 📈 Future Enhancements

* Integration with real wearable data (Fitbit, Apple Watch, etc.)
* Machine learning–based sleep quality prediction
* Web or mobile dashboard
* Export reports as PDF

---
