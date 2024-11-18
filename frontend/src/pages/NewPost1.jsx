import FrameComponent2 from "../components/FrameComponent2";
import SelectionArea from "../components/SelectionArea";
import "./NewPost1.css";

const NewPost1 = () => {
  return (
    <div className="new-post">
      <FrameComponent2 />
      <main className="content-area">
        <section className="posts">
          <div className="post-item">
            <SelectionArea rectangle="/rectangle@2x.png" emptySelection="1" />
            <SelectionArea
              selectionAreaBackgroundImage="unset"
              rectangle="/rectangle@2x.png"
              emptySelection="2"
            />
            <img
              className="selection-area-icon"
              loading="lazy"
              alt=""
              src="/rectangle@2x.png"
            />
            <img
              className="selection-area-icon"
              loading="lazy"
              alt=""
              src="/rectangle@2x.png"
            />
            <img
              className="selection-area-icon"
              loading="lazy"
              alt=""
              src="/rectangle@2x.png"
            />
            <SelectionArea
              selectionAreaBackgroundImage="unset"
              rectangle="/rectangle@2x.png"
              emptySelection="3"
            />
            <img
              className="selection-area-icon"
              loading="lazy"
              alt=""
              src="/rectangle@2x.png"
            />
            <img
              className="selection-area-icon"
              loading="lazy"
              alt=""
              src="/rectangle@2x.png"
            />
            <img
              className="selection-area-icon"
              loading="lazy"
              alt=""
              src="/rectangle@2x.png"
            />
            <img
              className="selection-area-icon"
              loading="lazy"
              alt=""
              src="/rectangle@2x.png"
            />
            <img
              className="selection-area-icon"
              loading="lazy"
              alt=""
              src="/rectangle@2x.png"
            />
            <img
              className="selection-area-icon"
              loading="lazy"
              alt=""
              src="/rectangle@2x.png"
            />
          </div>
          <div className="post-actions">
            <div className="action-item-one">
              <img
                className="selection-area-icon"
                loading="lazy"
                alt=""
                src="/rectangle@2x.png"
              />
              <img
                className="selection-area-icon"
                loading="lazy"
                alt=""
                src="/rectangle@2x.png"
              />
              <img
                className="selection-area-icon"
                loading="lazy"
                alt=""
                src="/rectangle-11@2x.png"
              />
            </div>
            <div className="action-item-two">
              <img
                className="action-shapes-two"
                loading="lazy"
                alt=""
                src="/rectangle-12@2x.png"
              />
              <img
                className="action-shapes-two"
                loading="lazy"
                alt=""
                src="/rectangle-12@2x.png"
              />
            </div>
          </div>
          <footer className="bars-home-indicator">
            <div className="background" />
            <div className="line" />
          </footer>
        </section>
      </main>
    </div>
  );
};

export default NewPost1;
