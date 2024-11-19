import FrameComponent5 from "../components/FrameComponent5";
import PostTop from "../components/PostTop";
import PostBottom from "../components/PostBottom";
import "./Main.css";

const Main = () => {
  return (
    <div className="main">
      <FrameComponent5 />
      <main className="post-area">
        <PostTop />
        <section className="photo-count">
          <img
            className="photo-count-background"
            loading="lazy"
            alt=""
            src="/rectangle@2x.png"
          />
          <div className="photo-number">
            <div className="rectangle" />
            <div className="div3">1/3</div>
          </div>
        </section>
        <PostBottom />
      </main>
    </div>
  );
};

export default Main;
