import FrameComponent3 from "../components/FrameComponent3";
import AddMusic from "../components/AddMusic";
import Recommendation from "../components/Recommendation";
import FrameComponent4 from "../components/FrameComponent4";
import "./NewPost2.css";

const NewPost2 = () => {
  return (
    <div className="new-post1">
      <section className="frame-parent">
        <FrameComponent3 />
        <div className="pictures-wrapper">
          <div className="pictures">
            <img
              className="pictures-child"
              loading="lazy"
              alt=""
              src="/rectangle-10@2x.png"
            />
            <img
              className="pictures-child"
              loading="lazy"
              alt=""
              src="/rectangle-111@2x.png"
            />
            <img
              className="pictures-child"
              loading="lazy"
              alt=""
              src="/rectangle-121@2x.png"
            />
          </div>
        </div>
        <div className="wrapper">
          <div className="div1">문구를 입력하세요...</div>
        </div>
      </section>
      <section className="add-music-parent">
        <div className="add-music">
          <AddMusic />
          <Recommendation />
        </div>
        <div className="music">
          <div className="icon-wrapper">
            <img className="icon" loading="lazy" alt="" src="/icon-2.svg" />
          </div>
          <div className="div2">혜원 - 마루는 강쥐</div>
        </div>
      </section>
      <FrameComponent4 />
    </div>
  );
};

export default NewPost2;
