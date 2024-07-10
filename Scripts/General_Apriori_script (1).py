import pandas as pd

class AprioriRecommender:
    def __init__(self, rules_path):
        self.rules = pd.read_csv(rules_path)

    def get_recommendations(self, items, top_n=5):
        recommendations = []

        for item in items:
            # Find rules where the item is in the antecedent
            matching_rules = self.rules[self.rules['antecedents'].apply(lambda x: item in eval(x))]

            # Sort matching rules by lift and support
            matching_rules = matching_rules.sort_values(by=['lift', 'support'], ascending=False)

            # Get top-n recommendations
            for _, rule in matching_rules.head(top_n).iterrows():
                consequents = eval(rule['consequents'])
                for consequent in consequents:
                    if consequent not in items and consequent not in recommendations:
                        recommendations.append(consequent)
                        if len(recommendations) >= top_n:
                            break
                if len(recommendations) >= top_n:
                    break

        return recommendations

# Example usage
if __name__ == "__main__":
    recommender = AprioriRecommender('/content/drive/My Drive/CAPSTONE/FIXED_DATASETS/df_filtered_apriori_rules.csv')
    items = ['item1', 'item2']  # Replace with actual items
    recommendations = recommender.get_recommendations(items)
    print(f"Recommendations for items {items}: {recommendations}")
