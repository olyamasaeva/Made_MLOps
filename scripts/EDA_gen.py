import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pdf_gen import pdf
import sys 
import io 
def get_df_stats(df: pd.DataFrame, doc):
    # Data exploration
    # Parameters:
    # df : pd.DataFrame - dataframe information about which we add to pdf
    # doc - document from which we form pdf
    
    doc.add_title("DataFrame fragment demonstration")
    doc.add_dataframe(df.head())
    doc.add_title("DataFrame columns information")
    buf = io.StringIO()
    df.info(buf=buf)
    s = buf.getvalue()
    doc.add_text(s)
    doc.add_title("DataFrame columns statistics")
    doc.add_dataframe(df.describe())

def print_sns_plot(df: pd.DataFrame, doc, hue_col_name :str, x_col_name : str, x_ticklabels :  list, title : str):
    # Prints to pdf a customize countplot
    # Parameters:
    # df : pd.DataFrame - Dataframe from which we plot
    # doc - document from which we form pdf
    # hue_col_name : str - column name from which we take hue for countplot
    # x_col_name : str - column name from which we take x for countplot
    # x_ticklabels : list of strings - labels of the counts
    # title : str - name of plot
    fig = plt.figure(figsize=(18,6))
    cp_condition_plot = None
    if hue_col_name != None:
        cp_condition_plot = sns.countplot(x=x_col_name, hue=hue_col_name, data=df)
    else:
        cp_condition_plot = sns.countplot(x = df[x_col_name])
    if x_ticklabels != None:
        cp_condition_plot.set_xticklabels(x_ticklabels)
    cp_condition_plot.set_title(title)
    doc.add_image(fig)
    plt.close(fig)

def print_hist_plot(df : pd.DataFrame, doc, x_col_name : str, y_col_name : str, title : str, legend : list):
    # Prints to pdf a customized hist plot
    # Parameters:
    # df : pd.DataFrame - Dataframe from which we plot
    # doc - document from which we form pdf
    # y_col_name : str - column name from which we group
    # x_col_name : str - column name for feature for hist
    # legend : list of strings - labels of legend of the plot
    # title : str - name of plot
    fig = plt.figure(figsize=(18,6))
    df.groupby(y_col_name)[x_col_name].plot(kind='hist')
    plt.legend(legend)
    plt.title(title)
    doc.add_image(fig)

def target_exploration(df : pd.DataFrame, target : pd.Series, doc):
    # Print to pdf target exploration
    # Parameters:
    # df: pd.dataFrame - dataframe from which target we explore
    # target : pd.Series - target column 
    # doc - document from which we form pdf
    print_sns_plot(df, doc, None, 'condition', None, "Target countplot")
    doc.add_dataframe(df['condition'].value_counts().to_frame())
    doc.add_text(f"targer 1's count in percentage is {round(target.value_counts()[1]/len(target), 2)} %")
    doc.add_text(f"target 0's count in percentage is {round(target.value_counts()[0]/target.shape[0], 2)} %")
    doc.add_text("Conclusion: dataset is unbalanced")

def distr_plot(df: pd.DataFrame, doc):    
    # Printing features distribution plots
    # Parameters:
    # df: pd.dataFrame - dataframe from which target we explore
    # doc - document from which we form pdf
    plots_logs = [("sns", "sex", ['Male', 'Female'], "Gender/disease frequency distribution", "Conclusion: Females has heart diesease more frequent than males"),
                  ("sns", "cp",  ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'], "chest pain type/disease frequency distribution", "conclusion most of patients with heart disease were asymptomatic"),
                  ("sns", "trestbps", None,  "resting blood pressure/disease frequency distribution", None),
                  ("hist", "age", ['does not have heart disease',' has heart disease'], "Age distribution",  "Conclusion: Eldery patient more frequent have heart disease"),
                  ("hist", "chol", ['does not have heart disease',' has heart disease'], "serum cholestora	distribution", None),
                  ("sns", "fbs", ['fbs <= 120 mg/dl','fbs > 120 mg/dl'], "fasting blood sugar/disease frequency distribution","Conclusion: people with low fbs more frequent have heart disease"),
                  ("sns", "restecg", ['normal','having ST-T wave abnormality','showing probable or definite left ventricular hypertrophy'], "resting electrocardiographic results/disease frequency distribution", "Conclusion: Patients showing left ventriculuar hupertrophy more frequent has heart disease" ),
                  ("hist", "chol", ['does not have heart disease',' has heart disease'], "maximum heart rate achieved", None),
                  ("sns", "exang", ['no','yes'], "exercise induced angina/condition frequency distribution","Conclusion: patients who had excercise included angine more frequent had heart disease"),
                  ("sns", "oldpeak", None,"ST depression induced by exercise relative to resta/condition frequency distribution","Conclusion: patients with high ST depression have more frequent heart disease"),
                  ("sns", "slope", ['normal',' ST-T wave abnormality','showing left ventricular hypertrophy'], "Slope/disease frequency distribution", "Conclusion: patients with flat slope more frequent have the heart disease"),
                  ("sns", "ca", None, "number of major vessels frequency distribution", "Conclusion: patients with more number of major vessels have more frequent heart disease"),
                  ("sns", "thal", ['normal','fixed defect','reversable defect'], "Thalassemia/condition frequency distribution", "Conclusion: patient with fixed or reversable defect more frequent have heart disease")]
    for type, x, x_ticks, title, comment in plots_logs:
        if type == "sns":
            print_sns_plot(df, doc, "condition", x, x_ticks, title)
        else:
            print_hist_plot(df, doc, x, "condition", title, x_ticks)
        if comment != None:
            doc.add_text(comment)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No data to read!")
        print("Use the script as 'python EDA_gen.py PATH_TO_FILE'")
        exit()
    file_name = sys.argv[1]
    df =  pd.read_csv(file_name)
    outputdir  = "report.pdf"
    if len(sys.argv) > 2:
        outputdir = sys.argv[2] + '/' + outputdir
    doc = pdf(outputdir, "EDA for Heart disease dataset")
    get_df_stats(df, doc)
    target = df['condition']
    features =  df.iloc[:,:-1]
    # Target exploration
    doc.add_title("Checking the target balance")
    target_exploration(df, target, doc)
    # Parameters distribution plots
    doc.add_title("Parameters distribution plots")
    distr_plot(df, doc)
    # Correlation heatmap
    doc.add_title("Correlation heatmap")
    fig = plt.figure(figsize=(18,6))
    corr = features.corr()
    sns.heatmap(corr, annot=True, vmin=-1.0,)
    plt.title("Features correlation heatmap")
    doc.add_image(fig)
    doc.add_text("Conclusion: No really strong correlation between any pair of variables")
    # outlier search
    doc.add_title("Outlier search")
    interesting_features=df[['age','trestbps','chol','thalach','oldpeak']]
    for column in interesting_features:
            fig  = plt.figure(figsize=(12,0.8))
            sns.boxplot(data=df, x=column, palette="Paired")
            doc.add_image(fig)
    doc.add_text("Conclusion: this dataset contains outliers")
    doc.generate_pdf()





