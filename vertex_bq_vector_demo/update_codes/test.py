# ================== Testing ==================
if __name__ == "__main__":
    # Example questions
    # question = "Order the gender and age group by the highest satisfaction level?"
    question = "Count number of responses by age?"

    state: AgentState = {
        "question": question,
        "history": [],
    }

    result = compiled_graph.invoke(state)

    print("\n\n========== FINAL RESULT (Markdown for chat) ==========\n")
    print(result.get("final_result", "No result"))

    print("\n========== TIMING ==========")
    print(result.get("timing", {}))

    df_res = result.get("df_result")
    if isinstance(df_res, pd.DataFrame):
        print("\n========== RAW DF RESULT (head) ==========")
        print(df_res.head())
    print("\n=================================\n")
