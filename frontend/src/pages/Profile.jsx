import TopBar from "../components/TopBar";
import FrameComponent from "../components/FrameComponent";
import "./Profile.css";

const Profile = () => {
  return (
    <div className="profile">
      <img className="tabs-icon" loading="lazy" alt="" src="/tabs@2x.png" />
      <section className="account-info-parent">
        <div className="account-info">
          <div className="mask" />
          <TopBar />
          <FrameComponent />
          <div className="bio-wrapper">
            <div className="bio">
              <b className="name">프로미</b>
              <div className="short-bio">나는야 프로메테우스의 얼굴🔥</div>
            </div>
          </div>
        </div>
        <img className="posts-icon" loading="lazy" alt="" src="/posts@2x.png" />
      </section>
      <img
        className="tab-bar-icon"
        loading="lazy"
        alt=""
        src="/tab-bar@2x.png"
      />
      <div className="bars-home-indicator2">
        <div className="background2" />
        <div className="bars-home-indicator3">
          <div className="line2" />
          <footer className="background2" />
          <div className="line3" />
        </div>
      </div>
    </div>
  );
};

export default Profile;
