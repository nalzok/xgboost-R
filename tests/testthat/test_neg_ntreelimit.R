require(xgboost)

context("predictions with negative ntreelimit")

data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test
set.seed(1994)

test_that("train and predict binary classification", {
  nrounds = 2
  expect_output(
    bst <- xgboost(data = train$data, label = train$label, max_depth = 2,
                  eta = 1, nthread = 2, nrounds = nrounds, objective = "binary:logistic")
  , "train-error")

  #######################
  # Predict probability #
  #######################

  pred <- predict(bst, test$data)
  expect_length(pred, 1611)

  pred_zero <- predict(bst, test$data, ntreelimit = 0)
  expect_equal(pred, pred_zero)
  pred_nrounds <- predict(bst, test$data, ntreelimit = nrounds)
  expect_equal(pred, pred_nrounds)

  pred_1 <- predict(bst, test$data, ntreelimit = 1)
  pred_neg_1 <- predict(bst, test$data, ntreelimit = -1)
  expect_equal(pred_1, pred_neg_1)

  # qlogis() is just logit()
  # see: https://ro-che.info/articles/2018-08-11-logit-logistic-r
  pred_nrounds_minus_one <- predict(bst, test$data, ntreelimit = nrounds - 1)
  pred_neg_nrounds <- predict(bst, test$data, ntreelimit = -nrounds)
  expect_equal(qlogis(pred_nrounds),
               qlogis(pred_nrounds_minus_one) + qlogis(pred_neg_nrounds),
               tolerance = 5e-6)


  #################################
  # Predict feature contributions #
  #################################

  pred <- predict(bst, test$data, predcontrib = TRUE)
  expect_equal(dim(pred), c(1611, 127))
  expect_equal(rowSums(pred), qlogis(predict(bst, test$data)),
               tolerance = 5e-6)

  pred_zero <- predict(bst, test$data, ntreelimit = 0,
                       predcontrib = TRUE)
  expect_equal(dim(pred), dim(pred_zero))
  expect_equal(pred, pred_zero)
  pred_nrounds <- predict(bst, test$data, ntreelimit = nrounds,
                          predcontrib = TRUE)
  expect_equal(dim(pred), dim(pred_nrounds))
  expect_equal(pred, pred_nrounds)

  pred_1 <- predict(bst, test$data, ntreelimit = 1,
                    predcontrib = TRUE)
  pred_neg_1 <- predict(bst, test$data, ntreelimit = -1,
                        predcontrib = TRUE)
  expect_equal(pred_1, pred_neg_1)

  pred_nrounds_minus_one <- predict(bst, test$data, ntreelimit = nrounds - 1,
                                    predcontrib = TRUE)
  pred_neg_nrounds <- predict(bst, test$data, ntreelimit = -nrounds,
                              predcontrib = TRUE)
  expect_equal(pred_nrounds, pred_nrounds_minus_one + pred_neg_nrounds,
               tolerance = 5e-6)


  #############################################
  # Predict approximate feature contributions #
  #############################################

  pred <- predict(bst, test$data, predcontrib = TRUE, approxcontrib = TRUE)
  expect_equal(dim(pred), c(1611, 127))
  expect_equal(rowSums(pred), qlogis(predict(bst, test$data)),
               tolerance = 5e-6)

  pred_zero <- predict(bst, test$data, ntreelimit = 0,
                       predcontrib = TRUE, approxcontrib = TRUE)
  expect_equal(dim(pred), dim(pred_zero))
  expect_equal(pred, pred_zero)
  pred_nrounds <- predict(bst, test$data, ntreelimit = nrounds,
                          predcontrib = TRUE, approxcontrib = TRUE)
  expect_equal(dim(pred), dim(pred_nrounds))
  expect_equal(pred, pred_nrounds)

  pred_1 <- predict(bst, test$data, ntreelimit = 1,
                    predcontrib = TRUE, approxcontrib = TRUE)
  pred_neg_1 <- predict(bst, test$data, ntreelimit = -1,
                        predcontrib = TRUE, approxcontrib = TRUE)
  expect_equal(pred_1, pred_neg_1)

  pred_nrounds_minus_one <- predict(bst, test$data, ntreelimit = nrounds - 1,
                                    predcontrib = TRUE, approxcontrib = TRUE)
  pred_neg_nrounds <- predict(bst, test$data, ntreelimit = -nrounds,
                              predcontrib = TRUE, approxcontrib = TRUE)
  expect_equal(pred_nrounds, pred_nrounds_minus_one + pred_neg_nrounds,
               tolerance = 5e-6)


  #############################################
  # Predict feature interaction contributions #
  #############################################

  pred <- predict(bst, test$data, predinteraction = TRUE)
  expect_equal(dim(pred), c(1611, 127, 127))

  pred_zero <- predict(bst, test$data, ntreelimit = 0,
                       predinteraction = TRUE)
  expect_equal(dim(pred), dim(pred_zero))
  expect_equal(pred, pred_zero)
  pred_nrounds <- predict(bst, test$data, ntreelimit = nrounds,
                          predinteraction = TRUE)
  expect_equal(dim(pred), dim(pred_nrounds))
  expect_equal(pred, pred_nrounds)

  pred_1 <- predict(bst, test$data, ntreelimit = 1,
                    predinteraction = TRUE)
  pred_neg_1 <- predict(bst, test$data, ntreelimit = -1,
                        predinteraction = TRUE)
  expect_equal(pred_1, pred_neg_1)

  pred_nrounds_minus_one <- predict(bst, test$data, ntreelimit = nrounds - 1,
                                    predinteraction = TRUE)
  pred_neg_nrounds <- predict(bst, test$data, ntreelimit = -nrounds,
                              predinteraction = TRUE)
  expect_equal(pred_nrounds, pred_nrounds_minus_one + pred_neg_nrounds,
               tolerance = 5e-6)
})

