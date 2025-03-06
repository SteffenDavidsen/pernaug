import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl


def rename_duplicates(columns):
    """Rename duplicate column names by appending a suffix."""
    counts = {}
    new_columns = []
    for col in columns:
        if pd.isna(col):
            col = ""
        if col in counts:
            counts[col] += 1
            new_columns.append(f"{col}_{counts[col]}")
        else:
            counts[col] = 0
            new_columns.append(col)
    return new_columns


def break_text(text, max_len=15):
    """Break long text into multiple lines."""
    return '\n'.join([text[i:i + max_len] for i in range(0, len(text), max_len)])


st.set_page_config(layout="wide")
st.title("Tactic Matrix Generator")

uploaded_file = st.file_uploader("Upload your file", type=["xls", "xlsx", "csv"])
if uploaded_file is not None:
    try:
        # Read the file as a tab-separated text file
        df = pd.read_csv(uploaded_file, header=None, sep="\t")
    except Exception as e:
        st.error("Could not read file: " + str(e))
    else:
        df = df.replace(r'^\s*$', pd.NA, regex=True)

        # --- Extract sections for Team A and Team B tactics ---
        team_a_index = df.index[
            df.apply(lambda row: row.astype(str).str.contains("Team A", case=False, na=False).any(), axis=1)
        ]
        team_b_index = df.index[
            df.apply(lambda row: row.astype(str).str.contains("Team B", case=False, na=False).any(), axis=1)
        ]

        if len(team_a_index) == 0:
            st.error("Could not find 'Team A' in the data.")
        elif len(team_b_index) == 0:
            st.error("Could not find 'Team B' in the data.")
        else:
            team_a_start = team_a_index[0]
            team_b_start = team_b_index[0]

            # Adjust extraction so that the first tactic for each team is included.
            section1 = df.loc[team_a_start + 1: team_b_start - 1]
            section2 = df.loc[team_b_start + 1:]

            # Extract (ID, name): ID from column A (index 0) and name from column N (index 13)
            team_A_data = [(row[0], row[13]) for idx, row in section1.iterrows() if not pd.isna(row[13])]
            team_B_data = [(row[0], row[13]) for idx, row in section2.iterrows() if not pd.isna(row[13])]

            # --- Extract the lookup table from the third section ---
            team1_index = df.index[
                df.apply(lambda row: row.astype(str).str.contains("Team 1", case=False, na=False).any(), axis=1)
            ]
            if len(team1_index) == 0:
                st.error("Could not find 'Team 1' in the data.")
            else:
                team1_start = team1_index[0]
                third_header = rename_duplicates(df.iloc[team1_start].tolist())
                third_data = df.loc[team1_start + 1:]
                third_df = pd.DataFrame(third_data.values, columns=third_header)

                # For the lookup table, assume:
                # - Column A (first) contains Team A tactic ID.
                # - Column C (third) contains Team B tactic ID.
                # - Column E (fifth) contains value E.
                # - Column F (sixth) contains value F.
                col_A = third_df.columns[0]
                col_C = third_df.columns[2]
                col_E = third_df.columns[4]
                col_F = third_df.columns[5]

                # Prepare display names for tactics and ensure they are unique.
                team_A_display = rename_duplicates([break_text(name) for (_, name) in team_A_data])
                team_B_display = rename_duplicates([break_text(name) for (_, name) in team_B_data])

                # Create the matrix with an extra "Weight" row at the top.
                matrix = pd.DataFrame(index=["Weight"] + team_A_display, columns=team_B_display)

                # --- Weight Input Section ---
                st.write("Enter weight for each Team B tactic (leave blank for zero):")
                cols = st.columns(len(team_B_display))
                weights = {}
                for col, tactic in zip(cols, team_B_display):
                    with col:
                        input_val = st.text_input(f"{tactic}", value="", key=tactic)
                        try:
                            weights[tactic] = round(float(input_val), 1) if input_val.strip() else 0.0
                        except ValueError:
                            weights[tactic] = 0.0
                total_weight = round(sum(weights.values()), 1)
                st.markdown(f"**Total Weight: {total_weight:.1f}**")
                if abs(total_weight - 100) > 0.1:
                    st.error("⚠️ The total weight must sum to 100!")
                else:
                    st.success("✅ The total weight is correctly set.")
                matrix.loc["Weight"] = [weights[tactic] for tactic in team_B_display]

                calculation_mode = st.radio(
                    "Select Calculation Mode",
                    options=["xP (Expected Points)", "Chance of not losing"],
                    index=0
                )

                # Fill in matrix cells based on the selected mode.
                for i, (id_A, _) in enumerate(team_A_data):
                    for j, (id_B, _) in enumerate(team_B_data):
                        match = third_df[(third_df[col_A] == id_A) & (third_df[col_C] == id_B)]
                        if match.empty:
                            matrix.loc[team_A_display[i], team_B_display[j]] = np.nan
                        else:
                            try:
                                val_E = float(match.iloc[0][col_E])
                            except:
                                val_E = 0.0
                            try:
                                val_F = float(match.iloc[0][col_F])
                            except:
                                val_F = 0.0
                            if calculation_mode == "xP (Expected Points)":
                                computed = (val_E * 3 / 100) + (val_F / 100)
                            else:
                                computed = (val_E) + (val_F)
                            matrix.loc[team_A_display[i], team_B_display[j]] = computed

                # --- Compute Extra "Total" Column (using an explicit loop) ---
                total_list = []
                for idx in matrix.index:
                    if idx == "Weight":
                        total_list.append(np.nan)
                    else:
                        row_total = 0.0
                        for col in team_B_display:
                            cell_val = matrix.loc[idx, col]
                            # If the cell returns a Series (due to duplicates), take the first element.
                            if isinstance(cell_val, pd.Series):
                                cell_val = cell_val.iloc[0]
                            if pd.isna(cell_val):
                                cell_val = 0.0
                            else:
                                cell_val = float(cell_val)
                            row_total += cell_val * (weights[col] / 100.0)
                        total_list.append(row_total)
                matrix["Total"] = total_list

                # --- Conditional Formatting ---
                non_weight_rows = [idx for idx in matrix.index if idx != "Weight"]
                numeric_values = []
                for idx in non_weight_rows:
                    for col in team_B_display:
                        try:
                            numeric_values.append(float(matrix.loc[idx, col]))
                        except:
                            pass
                vmin = min(numeric_values) if numeric_values else 0
                vmax = max(numeric_values) if numeric_values else 1

                total_numeric = []
                for idx in non_weight_rows:
                    try:
                        total_numeric.append(float(matrix.loc[idx, "Total"]))
                    except:
                        pass
                vmin_total = min(total_numeric) if total_numeric else 0
                vmax_total = max(total_numeric) if total_numeric else 1

                styled_matrix = (matrix.style
                                 .set_properties(**{'text-align': 'center'})
                                 .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
                                 .background_gradient(subset=pd.IndexSlice[non_weight_rows, team_B_display],
                                                      cmap="RdYlGn", vmin=vmin, vmax=vmax)
                                 .background_gradient(subset=pd.IndexSlice[non_weight_rows, "Total"],
                                                      cmap="RdYlGn", vmin=vmin_total, vmax=vmax_total)
                                 .format("{:.2f}", subset=[col for col in matrix.columns if col != "Weight"])
                                 .format("{:.1f}", subset=pd.IndexSlice["Weight", :])
                                 )

                st.markdown(styled_matrix.to_html(), unsafe_allow_html=True)
