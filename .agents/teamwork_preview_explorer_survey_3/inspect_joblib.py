import joblib

for name in ["boruta.pkl", "scaler.pkl", "svm_model.pkl"]:
    try:
        obj = joblib.load(name)
        print(f"\nLoaded {name} with joblib:")
        print("Type:", type(obj))
        if hasattr(obj, "support_"):
            print("Selected features count:", sum(obj.support_))
            print("Support shape:", obj.support_.shape)
        if hasattr(obj, "mean_"):
            print("Scaler mean shape:", obj.mean_.shape)
        if hasattr(obj, "classes_"):
            print("Classes:", obj.classes_)
        if hasattr(obj, "get_params"):
            print("Params:", obj.get_params())
        if hasattr(obj, "n_features_in_"):
            print("n_features_in_:", obj.n_features_in_)
    except Exception as e:
        print(f"Failed {name}:", e)
