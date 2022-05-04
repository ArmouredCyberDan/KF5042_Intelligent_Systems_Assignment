clc;
clear;

%Creates a variable to hold the file name
filename = "credit_risk_dataset.csv";

%Creates a table to hold the credit_risk_dataset
creditFile = readtable(filename,'TextType','String');
creditFile.loan_grade = [];
creditFile.cb_person_default_on_file = [];

%Swaps the class column with the last column in the table
creditFile = movevars(creditFile, 'loan_status', 'After', 'cb_person_cred_hist_length');

%Remove rows with missing data, this increases performance of the
%neural network despite moving quite a few rows
creditFile = rmmissing(creditFile);

%Creates a variable to hold the column header which denotes the class
%variable
labelName = "loan_status";

%Converts the class variable in to a category type
creditFile = convertvars(creditFile, labelName,'categorical');

%Creates a variable to hold the names of features with a categorical type
categoricalInputNames = ["person_home_ownership" "loan_intent"];
creditFile = convertvars(creditFile, categoricalInputNames, 'categorical');

for i = 1:numel(categoricalInputNames)
    name = categoricalInputNames(i);
    oh = onehotencode(creditFile(:,name));
    creditFile = addvars(creditFile,oh,'After',name);
    creditFile(:,name) = [];
end

creditFile = splitvars(creditFile);

%Normalises the numeric variables in the creditFile table
creditFile.person_age = normalize(creditFile.person_age);
creditFile.person_income = normalize(creditFile.person_income);
creditFile.person_emp_length = normalize(creditFile.person_emp_length);
creditFile.loan_amnt = normalize(creditFile.loan_amnt);
creditFile.loan_int_rate = normalize(creditFile.loan_int_rate);
creditFile.loan_percent_income = normalize(creditFile.loan_percent_income);
creditFile.cb_person_cred_hist_length = normalize(creditFile.cb_person_cred_hist_length);

%Creates a variable to hold all of the label variables, this needs to be a
%categorical type due to the method being used to create the ANN
classNames = categories(creditFile{:,labelName});

%Creates variables to hold the size of the total number of rows in the
%dataset and splits the amount in to train and test amount variables
numObservations = size(creditFile, 1);
numObservationsTrain = floor(0.85*numObservations);
numObservationsTest = numObservations - numObservationsTrain;

%Creates indexes to pick from when splitting the rows in to table variables
idx = randperm(numObservations);
idxTrain = idx(1:numObservationsTrain);
idxTest = idx(numObservationsTrain+1:end);

%Creates table variables to hold the split training/testing data
creditFileTrain = creditFile(idxTrain, :);
creditFileTest = creditFile(idxTest, :);

%Creates variables to hold the number of columns that are features and the
%column holding the class cells
numFeatures = size(creditFile, 2) - 1;
numClasses = numel(classNames);

%Layers are defined prior to creation of the ANN. This allows the user to
%tune the ANN should it be required
layers = [
    featureInputLayer(numFeatures,'Normalization', 'zscore')
    fullyConnectedLayer(20)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

miniBatchSize = 16;

%Options are defined prior to creation of the ANN. This allows the user to
%select operational parameters clearly before creation the ANN
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false);

%Creates a variable to hold the ANN. The network is trained on the training
%data partitioned above and the layers and options variables are designated
%in the constructor
net = trainNetwork(creditFileTrain, layers,options);

%Creates a variable and selects the ANN created above to be used to attempt
%to classify the data that was partioned as testing above
YPred = classify(net, creditFileTest,'MiniBatchSize',miniBatchSize);

%Creates a variable to hold the correct class cells for the outcome label
YTest = creditFileTest{:,labelName};

%Creates a variable and displays it. The variable takes a sum of the total
%number of predictions and actual outcome labels and divides it by the
%number of rows in the testing data.
accuracy = sum(YPred == YTest)/numel(YTest)
