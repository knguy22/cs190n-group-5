# CS190N Group 5

## Prerequisites

The following repository contains the (much of) the code that was used to perform our project. Because our project is evaluating the 
performance of NetMicroscope, part of the codebase was inextricably linked to NetMicrosopes', and we didn't seek permission 
to share it publically on Github. Thus, this codebase will not work as is; one must first contact the authors involved 
(Nick Feemster or Francesco Bronzino) to ask for the source code or obtain it in some other permissible way.

The NetMicroscope dataset can be obtained separately through this [public address](https://fbronzino.com/projects/vinference/).

The python dependencies will need to be installed too. Using a local virtual environment, install the dependencies located within `requirements.txt` using the appropriate command for your operating system.

## Training NetMicroscope

After obtaining the code, we aimed to replicate the NetMicroscope models. After separating the training and test data, run NetMicroscope on the training data in order to output the two models for resolution and startup delay.

## Training Trustee

Once the NetMicroscope models have been obtained alongside the training/test data split, we can finally use Trustee to evaluate its performance: 
1. Within `test_trustee.py`, edit the variables in the `if __name__ == "__main__"` section to point Trustee to the correct pkl files. 
2. Edit the Trustee regression and classifiation configuration (located within the `trustee_res` and `trustee_start` functions respectively) to the specifications you desire. The values we used are already set in `test_trustee.py`, so this step is not strictly required.

    ```py
    # Ex:
    trustee = ClassificationTrustee(expert=model)
    trustee.fit(X_train, y_train, num_iter=50, num_stability_iter=10, samples_size=0.2, verbose=True)

    # Ex:
    trustee = RegressionTrustee(expert=model)
    trustee.fit(X_train, y_train, num_iter=50, num_stability_iter=7, samples_size=0.2, verbose=True)
    ```

    Please consult the [Trustee documentation](https://trusteeml.github.io) for more details.
3. Run the file. Depending on your settings, this can take more than an hour. Also, make sure your system has enough RAM on hand.

Once these steps are completed, there should be two pdf files containing the pruned decision trees as well as console output of Trustee's performance and fidelity.

## Evaluating NetMicroscope

In order to test out some of the important decisions made by trustee, we created `filter_and_evaluate.py`:

1. Change the filenames in the functions `filter_and_eval_res` and `filter_and_eval_startup` to point to the correct models and test dataset.
2. Alter the conditions in the function `get_conditions` to match the nodes that you want to test. 
3. Run the code and observe the results.

