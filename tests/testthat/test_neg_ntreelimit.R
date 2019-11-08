require(xgboost)

context("predictions with negative ntreelimit")

data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test
set.seed(1994)

test_that("train and predict binary classification", {
  nrounds <- 2
  # TODO: investigate why things are broken when base_score is larger.
  base_score <- 0.7
  expect_output(
    bst <- xgboost(data = train$data, label = train$label, max_depth = 2,
                   eta = 1, nthread = 2, nrounds = nrounds,
                   base_score = base_score, objective = "binary:logistic")
  , "train-error")

  # See LogisticRegression::ProbToMargin
  # in src/src/objective/regression_loss.h
  predToRaw <- function(pred) {
      pred + log(1/base_score - 1)
  }

  # See LogisticRegression::PredTransform
  # in src/src/objective/regression_loss.h
  predUntransform <- function(pred) {
      # qlogis() is just logit()
      # see: https://ro-che.info/articles/2018-08-11-logit-logistic-r
      qlogis(pred)
  }

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

  pred_nrounds_minus_one <- predict(bst, test$data, ntreelimit = nrounds - 1)
  pred_neg_nrounds <- predict(bst, test$data, ntreelimit = -nrounds)
  expect_equal(predToRaw(predUntransform(pred_nrounds)),
               predToRaw(predUntransform(pred_nrounds_minus_one)) +
                   predToRaw(predUntransform(pred_neg_nrounds)),
               tolerance = 5e-6)


  #################################
  # Predict feature contributions #
  #################################

  pred <- predict(bst, test$data, predcontrib = TRUE)
  expect_equal(dim(pred), c(1611, 127))
  expect_equal(rowSums(pred), predUntransform(predict(bst, test$data)),
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
  expect_equal(pred_nrounds[, -127],
               pred_nrounds_minus_one[, -127] + pred_neg_nrounds[, -127],
               tolerance = 5e-6)


  #############################################
  # Predict approximate feature contributions #
  #############################################

  pred <- predict(bst, test$data, predcontrib = TRUE, approxcontrib = TRUE)
  expect_equal(dim(pred), c(1611, 127))
  expect_equal(rowSums(pred), predUntransform(predict(bst, test$data)),
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
  expect_equal(pred_nrounds[, -127],
               pred_nrounds_minus_one[, -127] + pred_neg_nrounds[, -127],
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
  expect_equal(pred_nrounds[, , -127],
               pred_nrounds_minus_one[, , -127] + pred_neg_nrounds[, , -127],
               tolerance = 5e-6)
})

test_that("train and predict regression", {
  nrounds = 2
  base_score <- 0.7
  expect_output(
    bst <- xgboost(data = train$data, label = train$label, max_depth = 2,
                   eta = 1, nthread = 2, nrounds = nrounds,
                   base_score = base_score, objective = "reg:squarederror")
  , "train-rmse")

  # See LinearSquareLoss::ProbToMargin
  # in src/src/objective/regression_loss.h
  predToRaw <- function(pred) {
      pred - base_score
  }

  # See LinearSquareLoss::PredTransform
  # in src/src/objective/regression_loss.h
  predUntransform <- function(pred) {
      pred
  }

  #################
  # Predict value #
  #################

  pred <- predict(bst, test$data)
  expect_length(pred, 1611)

  pred_zero <- predict(bst, test$data, ntreelimit = 0)
  expect_equal(pred, pred_zero)
  pred_nrounds <- predict(bst, test$data, ntreelimit = nrounds)
  expect_equal(pred, pred_nrounds)

  pred_1 <- predict(bst, test$data, ntreelimit = 1)
  pred_neg_1 <- predict(bst, test$data, ntreelimit = -1)
  expect_equal(pred_1, pred_neg_1)

  pred_nrounds_minus_one <- predict(bst, test$data, ntreelimit = nrounds - 1)
  pred_neg_nrounds <- predict(bst, test$data, ntreelimit = -nrounds)
  expect_equal(predToRaw(predUntransform(pred_nrounds)),
               predToRaw(predUntransform(pred_nrounds_minus_one)) +
                   predToRaw(predUntransform(pred_neg_nrounds)),
               tolerance = 5e-6)


  #################################
  # Predict feature contributions #
  #################################

  pred <- predict(bst, test$data, predcontrib = TRUE)
  expect_equal(dim(pred), c(1611, 127))
  expect_equal(rowSums(pred), predUntransform(predict(bst, test$data)),
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
  expect_equal(pred_nrounds[, -127],
               pred_nrounds_minus_one[, -127] + pred_neg_nrounds[, -127],
               tolerance = 5e-6)


  #############################################
  # Predict approximate feature contributions #
  #############################################

  pred <- predict(bst, test$data, predcontrib = TRUE, approxcontrib = TRUE)
  expect_equal(dim(pred), c(1611, 127))
  expect_equal(rowSums(pred), predUntransform(predict(bst, test$data)),
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
  expect_equal(pred_nrounds[, -127],
               pred_nrounds_minus_one[, -127] + pred_neg_nrounds[, -127],
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
  expect_equal(pred_nrounds[, , -127],
               pred_nrounds_minus_one[, , -127] + pred_neg_nrounds[, , -127],
               tolerance = 5e-6)
})
