# 📊 UPI Transaction Visualizer

Easily visualize and analyze your **Google Pay** and **Paytm** transaction data through insightful graphs and charts — all in one place.

---

## 🔍 Overview

This project helps you understand your digital transaction history by converting your raw **Google Pay** and **Paytm** data into clear and informative visualizations using Python.

You can run this project either locally or through:

1. **[Streamlit Web App](https://upidataviz.streamlit.app/)** - Instantly analyze your data with our online tool
2. **[Google Colab](https://colab.research.google.com/drive/1hzUANIiwNv-OyMVOBExpGEcljvjjNM3K#scrollTo=91470bdc&uniqifier=1)** — run in your browser

### 🌐 Using the Streamlit Web App

The [UPI Data Analyzer](https://upidataviz.streamlit.app/) provides an interactive web interface where you can:

- Upload your GPay and Paytm transaction files directly in your browser
- Get instant visualizations of your spending patterns
- View comprehensive transaction analysis including:
  - Total transactions, sent/paid amounts, and net flow
  - Overall transaction flow with intuitive pie and bar charts
  - Monthly spending trends
  - Transaction distribution analysis
  - Payment method usage statistics
  - Top recipients/senders
  - Daily spending patterns
- Download your processed data as CSV for further analysis

![UPI Data Analyzer Interface](https://raw.githubusercontent.com/ABoBo555/UPI-Data-Scrap-Viz/main/demo_images/app_interface.png)

No installation required - just visit the website and start analyzing your data! Your data remains private and is processed entirely in your browser session.

---

## 📝 Steps to Use

### 📥 1. Download Your Transaction Data

#### ✅ Google Pay
>1. Visit [Google Takeout](https://takeout.google.com/).
>2. Select only **Google Pay** and download your data.
>3. After downloading, unzip the folder.
>4. Navigate to: Takeout > Google Pay > My Activity > My Activity.html
>5. Upload the `My Activity.html` file into your project workspace.

#### ✅ Paytm
>1. Open the **Paytm App**.
>2. Go to: Paytm > Balance & History > ⋮ Menu > Download (.xlsx)
>3. Upload the downloaded `.xlsx` file into your project workspace.

---

### 🧪 2. Run the Python Files

In your local environment:

> Run this file run.py

This will load data and visualize your transaction data 📈
Then, you can see the graphs and charts saved in a new generated'Saved_charts' folder.

---

## 🚀 Try on Google Colab

You can run this directly from your browser using [Google Colab](https://colab.research.google.com):

👉 **[Open in Colab](https://colab.research.google.com/drive/1hzUANIiwNv-OyMVOBExpGEcljvjjNM3K#scrollTo=bf9b55fa&uniqifier=1)**

### In Colab:
1. When prompted(choose files), first upload the `My Activity.html` file (Google Pay).
2. On the next prompt, upload the **Paytm `.xlsx`** file.
3. Keep running the cells to see your transaction stats come to life!

---

## 📷 Generated Visuals include

- Net Overall Flow
- Usage of different Payemnt Methods
- Monthly trends
- Top spending months
- Transaction Distribution based-on range
- Top recipients/senders
- Meaningful Insight Card
---

## 🛠️ Tech Stack

- Python 🐍
- Pandas
- Matplotlib
- Seaborn
- Google Colab ☁️

---

## 🔧 Requirements

- Python 3.13.1
- Virtual environment recommended (`python -m venv myenv`)
- Install dependencies using:

```bash
pip install -r requirements.txt
```

## 🙌 Contribution & Feedback

Feel free to open issues, contribute enhancements, or suggest features. Let’s improve personal finance visibility for everyone! ✨

---
