conf.assignmentSlug = 'logistic-regression';
conf.itemName = 'Logistic Regression';
conf.partArrays = { ...
  { ...
    '1', ...
    { 'sigmoid.m' }, ...
    'Sigmoid Function', ...
  }, ...
  { ...
    '2', ...
    { 'costFunction.m' }, ...
    'Logistic Regression Cost', ...
  }, ...
  { ...
    '3', ...
    { 'costFunction.m' }, ...
    'Logistic Regression Gradient', ...
  }, ...
  { ...
    '4', ...
    { 'predict.m' }, ...
    'Predict', ...
  }, ...
  { ...
    '5', ...
    { 'costFunctionReg.m' }, ...
    'Regularized Logistic Regression Cost', ...
  }, ...
  { ...
    '6', ...
    { 'costFunctionReg.m' }, ...
    'Regularized Logistic Regression Gradient', ...
  }, ...
};
conf.output = @output;