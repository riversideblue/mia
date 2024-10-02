import createModel
import machineLearning
import sendModel

# ----- Main

if __name__ == "__main__":

    # --- Create foundation model
    foundation_model = createModel.main()

    # --- Send foundation model to Gateway
    sendModel.main(foundation_model)

    # --- Start machine learning
    machineLearning.main(foundation_model)