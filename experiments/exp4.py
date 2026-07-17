import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.metrics import accuracy_score



def run_knn():

    # Load Iris Dataset

    iris = load_iris(as_frame=True)


    # Select Features

    X = iris.data[
        [
            "sepal length (cm)",
            "sepal width (cm)"
        ]
    ]


    # Target

    y = iris.target



    # Split Dataset

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        stratify=y,

        random_state=0

    )



    # Create KNN Pipeline

    clf = Pipeline(

        steps=[

            (
                "scaler",
                StandardScaler()
            ),


            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=11
                )
            )

        ]

    )



    output = ""



    # Create Figure

    fig, axs = plt.subplots(

        ncols=2,

        figsize=(12,5)

    )



    # Try both weights

    for ax, weights in zip(

        axs,

        ("uniform","distance")

    ):


        # Set weight type

        clf.set_params(

            knn__weights=weights

        )


        # Train model

        clf.fit(

            X_train,

            y_train

        )



        # Prediction

        y_pred = clf.predict(

            X_test

        )



        accuracy = accuracy_score(

            y_test,

            y_pred

        )


        output += "\nKNN Weight : " + weights

        output += "\nAccuracy : " + str(round(accuracy,4))

        output += "\n"



        # Decision Boundary

        disp = DecisionBoundaryDisplay.from_estimator(

            clf,

            X_test,

            response_method="predict",

            plot_method="pcolormesh",

            xlabel="Sepal Length (cm)",

            ylabel="Sepal Width (cm)",

            shading="auto",

            alpha=0.5,

            ax=ax

        )



        scatter = disp.ax_.scatter(

            X.iloc[:,0],

            X.iloc[:,1],

            c=y,

            edgecolors="k"

        )



        disp.ax_.legend(

            scatter.legend_elements()[0],

            iris.target_names,

            loc="lower left",

            title="Classes"

        )



        disp.ax_.set_title(

            "KNN Classification\n" +

            "Weights = " +

            weights

        )



    plt.tight_layout()



    # Save graph

    plt.savefig(

        "static/graphs/knn.png",

        dpi=150,

        bbox_inches="tight"

    )


    plt.close()



    output += "\nDecision boundary graph generated successfully."


    return output