import { useEffect } from "react";
import {
  Routes,
  Route,
  useNavigationType,
  useLocation,
} from "react-router-dom";
import Profile from "./pages/Profile";
import NewPost1 from "./pages/NewPost1";
import NewPost2 from "./pages/NewPost2";
import NewPost from "./pages/NewPost";
import Main from "./pages/Main";

function App() {
  const action = useNavigationType();
  const location = useLocation();
  const pathname = location.pathname;

  useEffect(() => {
    if (action !== "POP") {
      window.scrollTo(0, 0);
    }
  }, [action, pathname]);

  useEffect(() => {
    let title = "";
    let metaDescription = "";

    switch (pathname) {
      case "/":
        title = "";
        metaDescription = "";
        break;
      case "/new-post":
        title = "";
        metaDescription = "";
        break;
      case "/new-post1":
        title = "";
        metaDescription = "";
        break;
      case "/new-post2":
        title = "";
        metaDescription = "";
        break;
      case "/main":
        title = "";
        metaDescription = "";
        break;
    }

    if (title) {
      document.title = title;
    }

    if (metaDescription) {
      const metaDescriptionTag = document.querySelector(
        'head > meta[name="description"]'
      );
      if (metaDescriptionTag) {
        metaDescriptionTag.content = metaDescription;
      }
    }
  }, [pathname]);

  return (
    <Routes>
      <Route path="/" element={<Profile />} />
      <Route path="/new-post" element={<NewPost1 />} />
      <Route path="/new-post1" element={<NewPost2 />} />
      <Route path="/new-post2" element={<NewPost />} />
      <Route path="/main" element={<Main />} />
    </Routes>
  );
}
export default App;
