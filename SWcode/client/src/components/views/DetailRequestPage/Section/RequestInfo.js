import React from 'react'
import { Collapse } from 'antd';
const { Panel } = Collapse;

function callback(key) {
    console.log(key);
  }

function RequestInfo(props) {



    return (
        <div>
            
            <Collapse defaultActiveKey={['1']} onChange={callback}>
                <Panel header="설명" key="1">
                    <p>{props.detail.description}</p>
                </Panel>
            </Collapse>


        </div>
    )
}

export default RequestInfo
