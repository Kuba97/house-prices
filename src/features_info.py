ID_COL = 'Id'
TARGET_COL = 'SalePrice'

NUMERIC_FEATS = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
           'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
           '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
           'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
           'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                 'MiscVal', 'MoSold', 'YrSold']

CATEGORIC_FEATS = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
             'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
             'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
             'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
             'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                   'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

NUMERIC_DISCRETE = ['BsmtHalfBath', 'HalfBath', 'FullBath', 'BsmtFullBath', 'Fireplaces', 'KitchenAbvGr', 'GarageCars',
                    'YrSold', 'BedroomAbvGr', 'OverallCond', 'OverallQual', 'TotRmsAbvGrd', 'MoSold', 'MSSubClass',
                    'YearRemodAdd', 'GarageYrBlt', 'YearBuilt']

NUMERIC_CONTINUOUS = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                      'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
                      'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

CATEGORIC_ORDINAL = ['Alley', 'BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'ExterCond',
                     'ExterQual', 'Fence', 'FireplaceQu', 'Functional', 'GarageCond', 'GarageQual', 'HeatingQC',
                     'KitchenQual', 'PavedDrive', 'Utilities', 'LotShape', 'LandSlope', 'Electrical', 'PoolQC',
                     'GarageFinish']

CATEGORIC_NOMINAL = ['MSZoning', 'Street', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
                     'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                     'Foundation', 'Heating', 'CentralAir', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']

FEAT_DICT = {'numeric': NUMERIC_FEATS, 'categoric': CATEGORIC_FEATS, 'numeric_discrete': NUMERIC_DISCRETE,
             'numeric_continuous': NUMERIC_CONTINUOUS, 'categoric_ordinal': CATEGORIC_ORDINAL,
             'categoric_nominal': CATEGORIC_NOMINAL}
