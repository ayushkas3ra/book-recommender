import pickle

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

# loading data
df = pickle.load(open("model/df.pkl", "rb"))
tfidf_matrix = pickle.load(open("model/tfidf_matrix.pkl", "rb"))
indices = pickle.load(open("model/indices.pkl", "rb"))
df_popular = pickle.load(open("model/df_popular.pkl", "rb"))

app = Flask(__name__)


# Routes
@app.route("/")
def index():
    return render_template(
        "home.html",
        books_names=df_popular["title"].tolist(),
        authors=df_popular["authors"].tolist(),
        images=df_popular["thumbnail"].tolist(),
        votes=df_popular["ratings_count"].tolist(),
        ratings=df_popular["average_rating"].tolist(),
        genres=df_popular["categories"].tolist(),
        ids=df_popular.index.tolist(),
    )


@app.route("/recommend_books", methods=["POST"])
def recommend():
    user_input = request.form.get("user_input")

    if user_input not in indices:
        return render_template("recommend.html", data=None, error="Book not found.")

    try:
        idx = indices[user_input]
        sim_score = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

        similar_idx = sim_score.argsort()[::-1][1:11]

        recommended_books = df.iloc[similar_idx]

        data = []
        for i in range(len(recommended_books)):
            temp_df = recommended_books.iloc[i]

            item = [
                temp_df["title"] if "title" in temp_df else "Unknown",
                temp_df["authors"] if "authors" in temp_df else "Unknown Author",
                temp_df["categories"] if "categories" in temp_df else "N/A",
                temp_df["average_rating"] if "average_rating" in temp_df else 0,
                temp_df["ratings_count"] if "ratings_count" in temp_df else 0,
                temp_df["thumbnail"] if "thumbnail" in temp_df else "",
                recommended_books.index[i],
            ]
            data.append(item)

        return render_template("recommend.html", data=data)

    except Exception as e:
        print(f"Error during recommendation: {e}")
        return render_template("recommend.html", data=None, error="An error occurred.")


@app.route("/book/<int:book_id>")
def book(book_id):
    book = df.iloc[book_id]
    book_details = {
        "title": book["title"],
        "author": book["authors"],
        "category": book["categories"],
        "average_rating": book["average_rating"],
        "ratings_count": book["ratings_count"],
        "thumbnail": book["thumbnail"],
        "description": book["description"],
        "id": book_id,
    }
    return render_template("book.html", book=book_details)


@app.route("/signin")
def signin():
    return render_template("signin.html")


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question")
        bookId = data.get("bookId")

        book = df.iloc[int(bookId)]
        title = book.get("title", "Unknown title")
        authors = book.get("authors", "Unknown author")
        categories = book.get("category", "Unknown")
        description = book.get("description", "No description available for this book")

        context = f"""
        Book title:{title}
        Author(s):{authors}
        Categories:{categories}
        Description:{description}
        """

        promptTemplate = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI librarian which is created by 'Ayush Kasera' and you will answer user question in a two lines strictly based on given context of the book, if you can not find information in the context given then you must simply say,'I do not know'.",
                ),
                ("human", "Book information:{context}\n\nUser Question: {question}"),
            ]
        )

        chain = promptTemplate | llm

        response = chain.invoke({"context": context, "question": question})

        return jsonify({"answer": response.content})

    except Exception as e:
        print(f"Error in /ask route: {e}")
        return jsonify({"answer": "Sorry, an internal error occurred."}), 500


if __name__ == "__main__":
    app.run(debug=True)
