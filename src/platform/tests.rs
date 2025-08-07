// src/platform/tests.rs

use super::actions::PlatformAction;
use super::backends::{BackendEvent, KeySymbol, Modifiers};
use super::mock::MockPlatform;
use super::platform_trait::Platform;
use super::PlatformEvent;
use anyhow::Result;

#[test]
fn it_should_create_mock_platform_without_panicking() {
    let (_platform, _state) =
        <MockPlatform as Platform>::new(80, 24, "sh".to_string(), vec!["-c".to_string(), "ls".to_string()])
            .unwrap();
}

#[test]
fn it_should_poll_events_that_were_pushed_to_the_mock_platform() -> Result<()> {
    let (mut platform, _state) =
        <MockPlatform as Platform>::new(80, 24, "sh".to_string(), vec!["-c".to_string(), "ls".to_string()])?;

    let key_event = BackendEvent::Key {
        symbol: KeySymbol::Char('a'),
        modifiers: Modifiers::empty(),
        text: "a".to_string(),
    };
    let platform_event = PlatformEvent::BackendEvent(key_event);

    platform.push_event(platform_event);

    let polled_events = platform.poll_events()?;
    assert_eq!(polled_events.len(), 1);

    if let PlatformEvent::BackendEvent(BackendEvent::Key { symbol, .. }) = &polled_events[0] {
        assert_eq!(*symbol, KeySymbol::Char('a'));
    } else {
        panic!("Unexpected event type");
    }
    Ok(())
}

#[test]
fn it_should_record_actions_dispatched_to_the_mock_platform() -> Result<()> {
    let (mut platform, _state) =
        <MockPlatform as Platform>::new(80, 24, "sh".to_string(), vec!["-c".to_string(), "ls".to_string()])?;

    let action = PlatformAction::Write("hello".as_bytes().to_vec());
    platform.dispatch_actions(vec![action])?;

    let dispatched_actions = platform.dispatched_actions();
    assert_eq!(dispatched_actions.len(), 1);

    if let PlatformAction::Write(data) = &dispatched_actions[0] {
        assert_eq!(data, "hello".as_bytes());
    } else {
        panic!("Unexpected action type");
    }

    Ok(())
}

#[test]
fn it_should_handle_a_resize_event_and_dispatch_a_resize_action() -> Result<()> {
    let (mut platform, _state) =
        <MockPlatform as Platform>::new(80, 24, "sh".to_string(), vec!["-c".to_string(), "ls".to_string()])?;

    let resize_event = BackendEvent::Resize {
        width_px: 1024,
        height_px: 768,
    };
    let platform_event = PlatformEvent::BackendEvent(resize_event);
    platform.push_event(platform_event);

    // In a real application, some component would poll the event and dispatch an action.
    // For this test, we'll simulate that behavior.
    let polled_events = platform.poll_events()?;
    for event in polled_events {
        if let PlatformEvent::BackendEvent(BackendEvent::Resize { width_px, height_px }) = event {
            let resize_action = PlatformAction::ResizePty {
                cols: width_px as u16 / 8,  // Assuming font width of 8px
                rows: height_px as u16 / 16, // Assuming font height of 16px
            };
            platform.dispatch_actions(vec![resize_action])?;
        }
    }

    let dispatched_actions = platform.dispatched_actions();
    assert_eq!(dispatched_actions.len(), 1);

    if let PlatformAction::ResizePty { cols, rows } = &dispatched_actions[0] {
        assert_eq!(*cols, 128);
        assert_eq!(*rows, 48);
    } else {
        panic!("Unexpected action type");
    }

    Ok(())
}
