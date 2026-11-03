"""Entry point of the interface: `streamlit run src/hotel_cancellation/ui/app.py`.

It only sets up the page, the navigation and the state of the API. Each page lives in `views/`,
not in `pages/`: Streamlit would read that folder as its older kind of navigation and warn about
a page it cannot find when a page is opened from its address.
"""

import streamlit as st

from hotel_cancellation.ui import layout, results

st.set_page_config(
    page_title="Cancelación de reservas de hotel",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

navigation = st.navigation(
    [
        st.Page("views/overview.py", title="Resumen", url_path="resumen", default=True),
        st.Page("views/prediction.py", title="Predicción", url_path="prediccion"),
        st.Page("views/evaluation.py", title="Evaluación", url_path="evaluacion"),
        st.Page("views/threshold.py", title="Umbral y coste", url_path="umbral"),
        st.Page("views/interpretation.py", title="Interpretabilidad", url_path="interpretabilidad"),
        st.Page("views/data.py", title="Datos", url_path="datos"),
    ]
)

layout.check_api()
layout.api_status(st.sidebar, verbose=False)
st.sidebar.caption("Solo la página de predicción necesita la API.")

missing = results.missing_results()
if missing:
    st.error(
        "Faltan resultados guardados que estas páginas necesitan:\n\n- " + "\n- ".join(missing)
    )
    st.stop()

navigation.run()
