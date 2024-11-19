import PropTypes from "prop-types";
import "./FrameComponent.css";

const FrameComponent = ({ className = "" }) => {
  return (
    <div className={`account-info-inner ${className}`}>
      <div className="photo-parent">
        <img className="photo-icon" loading="lazy" alt="" src="/photo.svg" />
        <div className="stats-wrapper">
          <div className="stats">
            <div className="stats-list">
              <a className="post-count">5</a>
              <a className="post-count">0</a>
              <a className="post-count">0</a>
            </div>
            <div className="stats-list1">
              <div className="followers">게시물</div>
              <div className="followers">팔로워</div>
              <div className="followers">팔로잉</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

FrameComponent.propTypes = {
  className: PropTypes.string,
};

export default FrameComponent;
