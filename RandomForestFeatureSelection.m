clc;
clear;

%Creates a variable to hold the file name
filename = "credit_risk_dataset.csv";

%Creates a table to hold the credit_risk_dataset
creditFile = readtable(filename,'TextType','String');

%Swaps the class column with the last column in the table
creditFile = movevars(creditFile, 'loan_status', 'After', 'cb_person_cred_hist_length');

%Remove rows with missing data, this increases performance of the
%neural network despite removing quite a few rows
creditFile = rmmissing(creditFile);

%Creates a variable to hold the outcome labels or classes
outcome = creditFile.loan_status;

%Deletes the loan_status column to prevent any confusion in the system this
creditFile.loan_status = [];

%Creates a variable to hold the predictor features of the creditfile table
categoricalInputNames = ["person_home_ownership" "cb_person_default_on_file" "loan_grade" "loan_intent"];

%Convert all categorical data, i.e. non-binary, non-numeric data, in to
%categories.
creditFile = convertvars(creditFile, categoricalInputNames, 'categorical');

for i = 1:numel(categoricalInputNames)
    name = categoricalInputNames(i);
    oh = onehotencode(creditFile(:,name));
    creditFile = addvars(creditFile,oh,'After',name);
    creditFile(:,name) = [];
end

creditFile = splitvars(creditFile);

%Creates the random forest 
t = templateTree('NumVariablesToSample','all',...
    'PredictorSelection','interaction-curvature','Surrogate','on');
rng(1);
Mdl = fitrensemble(creditFile, outcome,'Method','Bag','NumLearningCycles',200, ...
    'Learners',t);

%Creates a plot for visualisation of the importance rating for predictor
%features from the dataset
impOOB = oobPermutedPredictorImportance(Mdl);

figure
bar(impOOB)
title('Unbiased Predictor Importance Estimates')
xlabel('Predictor variable')
ylabel('Importance')
h = gca;
h.XTickLabel = Mdl.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';
set(h, "XTick", 1:26);