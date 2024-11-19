import PropTypes from "prop-types";
import "./PostTop.css";

const PostTop = ({ className = "" }) => {
  return (
    <div className={`post-top ${className}`}>
      <div className="rectangle1" />
      <div className="post-header">
        <img
          className="image-icon1"
          loading="lazy"
          alt=""
          src="/image1@2x.png"
        />
        <div className="profile-info">
          <a className="name1">promi123</a>
          <div className="music-info">
            <div className="music-icon-area">
              <img className="icon5" loading="lazy" alt="" src="/icon.svg" />
            </div>
            <div className="music3">로제, Bruno Mars - APT.</div>
          </div>
        </div>
      </div>
      <div className="more-icon-area">
        <img
          className="more-icon"
          loading="lazy"
          alt=""
          src="/more-icon@2x.png"
        />
      </div>
    </div>
  );
};

PostTop.propTypes = {
  className: PropTypes.string,
};

export default PostTop;
