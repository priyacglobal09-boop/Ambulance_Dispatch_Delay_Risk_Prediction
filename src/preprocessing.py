import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

class AmbulanceDataPipeline:
    """
    A robust, production-ready preprocessing pipeline for the Ambulance Dispatch Delay 
    Risk Prediction system. Handles data validation, stratified splitting, feature scaling,
    and multiple encoding approaches (One-Hot and Integer encoding) for different model types.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        # Define features
        self.categorical_cols = [
            'priority_level', 
            'incident_type', 
            'time_of_day', 
            'weather_conditions', 
            'day_of_week', 
            'traffic_density', 
            'dispatch_zone'
        ]
        self.numerical_cols = [
            'distance_to_scene', 
            'crew_experience_years', 
            'ambulance_age_years', 
            'temperature', 
            'historical_zone_delay_rate', 
            'caller_stress_score'
        ]
        self.target_col = 'delay_risk'
        self.case_id_col = 'case_id'
        
        # Properties populated after fitting
        self.categorical_dims = {}  # Dictionary storing {col_name: num_unique_categories}
        self.category_mappings = {} # Dictionary storing {col_name: list_of_categories}
        self.linear_feature_names = []
        self.is_fitted = False

    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """Loads and validates the csv file ensuring all required columns exist."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at: {file_path}")
            
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_cols = self.categorical_cols + self.numerical_cols
        if self.target_col in df.columns:
            required_cols = required_cols + [self.target_col]
            
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")
            
        # Drop duplicates and handle missing values (if any)
        df = df.drop_duplicates(subset=[self.case_id_col] if self.case_id_col in df.columns else None)
        df = df.dropna(subset=required_cols)
        
        return df

    def split_dataset(self, df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
        """Performs a stratified train/val/test split."""
        # Train and Temp (Val + Test)
        temp_size = test_size + val_size
        train_df, temp_df = train_test_split(
            df, 
            test_size=temp_size, 
            random_state=random_state, 
            stratify=df[self.target_col] if self.target_col in df.columns else None
        )
        
        # Split Temp into Val and Test
        val_ratio_of_temp = val_size / temp_size
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_of_temp),
            random_state=random_state,
            stratify=temp_df[self.target_col] if self.target_col in temp_df.columns else None
        )
        
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def fit(self, df_train: pd.DataFrame):
        """Fits all preprocessing transformers on the training dataframe."""
        # 1. Fit numerical scaler
        self.scaler.fit(df_train[self.numerical_cols])
        
        # 2. Fit one-hot encoder
        self.one_hot_encoder.fit(df_train[self.categorical_cols])
        
        # Save linear model feature names
        oh_feature_names = self.one_hot_encoder.get_feature_names_out(self.categorical_cols)
        self.linear_feature_names = self.numerical_cols + list(oh_feature_names)
        
        # 3. Fit ordinal/integer encoder
        self.ordinal_encoder.fit(df_train[self.categorical_cols])
        
        # Store dimensions & mappings for embedding layer configuration
        for i, col in enumerate(self.categorical_cols):
            cats = self.ordinal_encoder.categories_[i]
            self.category_mappings[col] = list(cats)
            # Add 1 for unknown tokens (which map to index 0)
            self.categorical_dims[col] = len(cats) + 1
            
        self.is_fitted = True
        return self

    def transform_linear(self, df: pd.DataFrame):
        """
        Transforms features for Linear/Logistic Regression.
        Standardizes numerical columns and One-Hot encodes categorical columns.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transforming.")
            
        X_num = self.scaler.transform(df[self.numerical_cols])
        X_cat = self.one_hot_encoder.transform(df[self.categorical_cols])
        
        X_combined = np.hstack([X_num, X_cat])
        
        if self.target_col in df.columns:
            y = df[self.target_col].values
            return X_combined, y
        return X_combined

    def transform_integer(self, df: pd.DataFrame):
        """
        Transforms features for neural networks (FCNN Integer and FCNN Embedding).
        Standardizes numerical columns and integer encodes categorical columns.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transforming.")
            
        X_num = self.scaler.transform(df[self.numerical_cols])
        
        # OrdinalEncoder outputs 0 to N-1, with -1 for unknowns.
        # Shift everything by +1 so that:
        # Unknown -> 0
        # Categories -> 1 to N
        X_cat = self.ordinal_encoder.transform(df[self.categorical_cols]) + 1
        
        # Combine numerical and categorical columns
        X_combined = np.hstack([X_num, X_cat])
        
        if self.target_col in df.columns:
            y = df[self.target_col].values
            return X_combined, y
        return X_combined

    def transform_single(self, input_dict: dict, model_type: str) -> np.ndarray:
        """
        Transforms a single raw prediction input (e.g. from the web app or an API).
        
        Parameters:
            input_dict (dict): Keys must match self.categorical_cols and self.numerical_cols.
            model_type (str): Either 'linear', 'integer_fcnn', or 'embedding_fcnn'.
            
        Returns:
            np.ndarray: Preprocessed features vector ready for inference.
        """
        # Convert single sample dict to a 1-row DataFrame
        df_single = pd.DataFrame([input_dict])
        
        if model_type == 'linear':
            return self.transform_linear(df_single)
        elif model_type in ['integer_fcnn', 'embedding_fcnn']:
            return self.transform_integer(df_single)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def save_pipeline(self, path: str):
        """Saves the fitted pipeline object to disk using pickle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load_pipeline(path: str) -> 'AmbulanceDataPipeline':
        """Loads a saved pipeline object from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)

if __name__ == '__main__':
    print("Testing Preprocessing Pipeline...")
    # Generate temporary synthetic data for verification
    from create_synthetic_data import generate_synthetic_data
    df = generate_synthetic_data(100)
    
    pipeline = AmbulanceDataPipeline()
    train_df, val_df, test_df = pipeline.split_dataset(df)
    pipeline.fit(train_df)
    
    X_lin, y_lin = pipeline.transform_linear(test_df)
    X_int, y_int = pipeline.transform_integer(test_df)
    
    print("Linear Transformation shape:", X_lin.shape)
    print("Integer Transformation shape:", X_int.shape)
    print("Fitted Categorical dimensions:", pipeline.categorical_dims)
    print("Pipeline code executed and verified successfully!")
