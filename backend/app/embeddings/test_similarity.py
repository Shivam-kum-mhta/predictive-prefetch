from embeddings import NewsEmbeddings

news_embeddings = NewsEmbeddings()

title1ID = "N33276"
title2ID = "N20483"
title3ID = "N7004"
title4ID = "N2186"
title5ID = "N36444"

similarity1 = news_embeddings.SimilarityBetweenTitles(title1ID, title2ID)
print(f"Similarity between {title1ID} and {title2ID}: {similarity1}")

similarity2 = news_embeddings.SimilarityBetweenTitles(title1ID, title3ID)
print(f"Similarity between {title1ID} and {title3ID}: {similarity2}")

similarity3 = news_embeddings.SimilarityBetweenTitles(title1ID, title4ID)
print(f"Similarity between {title1ID} and {title4ID}: {similarity3}")

similarity4 = news_embeddings.SimilarityBetweenTitles(title1ID, title5ID)
print(f"Similarity between {title1ID} and {title5ID}: {similarity4}")


similarity1 = news_embeddings.SimilarityBetweenAbstracts(title1ID, title2ID)
print(f"Similarity between abstract of {title1ID} and {title2ID}: {similarity1}")

similarity2 = news_embeddings.SimilarityBetweenAbstracts(title1ID, title3ID)
print(f"Similarity between abstract of {title1ID} and {title3ID}: {similarity2}")

similarity3 = news_embeddings.SimilarityBetweenAbstracts(title1ID, title4ID)
print(f"Similarity between abstract of {title1ID} and {title4ID}: {similarity3}")

similarity4 = news_embeddings.SimilarityBetweenAbstracts(title1ID, title5ID)
print(f"Similarity between abstract of {title1ID} and {title5ID}: {similarity4}")